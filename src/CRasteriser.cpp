//
// Copyright (C) 2017 Alex Smith
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//

#include "CRasteriser.h"
#include "CSurface.h"

#include <algorithm>
#include <cmath>

// Number of bits of sub-pixel precision. Snapping vertex positions to integers can cause noticable
// artifacts, particularly with animation. Therefore, we snap to sub-pixel positions instead. We
// use fixed-point arithmetic internally. Currently using 28.4, which is good for vertex positions
// ranging from [-2048, 2047].
static constexpr CRasteriser::Fixed kSubPixelBits = 4;
static constexpr CRasteriser::Fixed kSubPixelStep = 1 << kSubPixelBits;
static constexpr CRasteriser::Fixed kSubPixelMask = kSubPixelStep - 1;

// We're following D3D10/GL conventions and have pixel centres at .5 offsets, i.e. position
// (0.5, 0.5) corresponds exactly to the top left pixel.
static constexpr float kPixelCenter = 0.5f;

void CRasteriser::DrawTriangle(CSurface&      inSurface,
                               const SVertex* inVertices)
{
    // Window coordinate transformation parameters. TODO: Allow viewport to be specified.
    const float halfWindowWidth  = static_cast<float>(inSurface.GetWidth()) / 2;
    const float halfWindowHeight = static_cast<float>(inSurface.GetHeight()) / 2;
    const float windowCentreX    = halfWindowWidth;
    const float windowCentreY    = halfWindowHeight;

    // Set up internal vertex data.
    auto SetUpVertex =
        [&] (STriVertex& outVertex, const size_t inIndex)
        {
            const SVertex& sourceVertex = inVertices[inIndex];

            // Convert homogeneous to NDC coordinates.
            float x = sourceVertex.position.x / sourceVertex.position.w;
            float y = sourceVertex.position.y / sourceVertex.position.w;

            // Convert to window coordinates. We follow D3D conventions: clip/NDC space has Y
            // pointing up, but window coordinates have it down. Therefore, invert Y here.
            x = (x * halfWindowWidth)   + windowCentreX;
            y = (-y * halfWindowHeight) + windowCentreY;

            // To deal with the pixel centre, just bias the positions to have the centre at exact
            // integer coordinates.
            x -= kPixelCenter;
            y -= kPixelCenter;

            // Convert to fixed point.
            outVertex.x = lround(x * static_cast<float>(kSubPixelStep));
            outVertex.y = lround(y * static_cast<float>(kSubPixelStep));

            outVertex.index = inIndex;
        };

    STriVertex v0; SetUpVertex(v0, 0);
    STriVertex v1; SetUpVertex(v1, 1);
    STriVertex v2; SetUpVertex(v2, 2);

    // Determine winding of the vertices, by calculating the barycentric weight of v2 - if it is
    // positive, then the winding is CCW.
    const Fixed weight        = ((v0.x - v1.x) * (v2.y - v0.y)) - ((v0.y - v1.y) * (v2.x - v0.x));
    const ETriWinding winding = (weight < 0) ? kTriWinding_CW
                              : (weight > 0) ? kTriWinding_CCW
                                             : kTriWinding_Degenerate;

    if (winding == kTriWinding_CW)
    {
        // The main loop below assumes CCW, so swap v1 and v2.
        std::swap(v1, v2);
    }
    else if (winding == kTriWinding_Degenerate)
    {
        // Cull degenerate triangles.
        return;
    }

    // Follow the D3D/GL top-left fill rule: pixel centres on an edge are considered to be inside a
    // triangle if that edge is a top or left edge. See:
    // https://msdn.microsoft.com/en-us/library/windows/desktop/cc627092(v=vs.85).aspx#Triangle
    //
    // At this point the triangle is CCW. In a CCW triangle, a top edge is one that is exactly
    // horizontal, and goes to the left. A left edge is just one that goes down.
    //
    // We implement the fill rule by applying a bias to the calculated weights of each pixel in the
    // main loop based on whether the edge used to calculate that weight is top/left. The weight is
    // 0 if the pixel is on the edge, so biasing this by -1 for edges which aren't top/left will
    // exclude pixels which lie on it.
    auto IsTopLeft =
        [] (const STriVertex& inA, const STriVertex& inB)
        {
            const bool top  = inA.y == inB.y && inB.x < inA.x;
            const bool left = inB.y > inA.y;
            return top || left;
        };

    const Fixed bias0 = (IsTopLeft(v1, v2)) ? 0 : -1;
    const Fixed bias1 = (IsTopLeft(v2, v0)) ? 0 : -1;
    const Fixed bias2 = (IsTopLeft(v0, v1)) ? 0 : -1;

    // Calculate the bounding box of the triangle, rounded to whole pixels (min rounds down, max
    // rounds up).
    Fixed minX = std::min(v0.x, std::min(v1.x, v2.x)) & ~kSubPixelMask;
    Fixed minY = std::min(v0.y, std::min(v1.y, v2.y)) & ~kSubPixelMask;
    Fixed maxX = (std::max(v0.x, std::max(v1.x, v2.x)) + kSubPixelMask) & ~kSubPixelMask;
    Fixed maxY = (std::max(v0.y, std::max(v1.y, v2.y)) + kSubPixelMask) & ~kSubPixelMask;

    // Clip to the surface area.
    minX = std::max(minX, static_cast<Fixed>(0));
    minY = std::max(minY, static_cast<Fixed>(0));
    maxX = std::min(maxX, (static_cast<Fixed>(inSurface.GetWidth()) - 1) << kSubPixelBits);
    maxY = std::min(maxY, (static_cast<Fixed>(inSurface.GetHeight()) - 1) << kSubPixelBits);

    // At each pixel, the barycentric weights are given by:
    //
    //   w0 = (x * (v2.y - v1.y)) + (y * (v1.x - v2.x)) + ((v2.x * v1.y) - (v2.y * v1.x))
    //   w1 = (x * (v0.y - v2.y)) + (y * (v2.x - v0.x)) + ((v0.x * v2.y) - (v0.y * v2.x))
    //   w2 = (x * (v1.y - v0.y)) + (y * (v0.x - v1.x)) + ((v1.x * v0.y) - (v1.y * v0.x))
    //
    // The increment in the weight at each step in the X direction is given as follows:
    const Fixed xStep0 = (v2.y - v1.y) << kSubPixelBits;
    const Fixed xStep1 = (v0.y - v2.y) << kSubPixelBits;
    const Fixed xStep2 = (v1.y - v0.y) << kSubPixelBits;

    // The same for each step in the Y direction:
    const Fixed yStep0 = (v1.x - v2.x) << kSubPixelBits;
    const Fixed yStep1 = (v2.x - v0.x) << kSubPixelBits;
    const Fixed yStep2 = (v0.x - v1.x) << kSubPixelBits;

    // Weight at the start of the first row:
    Fixed w0Row = ((minY >> kSubPixelBits) * yStep0) + ((minX >> kSubPixelBits) * xStep0) + ((v2.x * v1.y) - (v2.y * v1.x));
    Fixed w1Row = ((minY >> kSubPixelBits) * yStep1) + ((minX >> kSubPixelBits) * xStep1) + ((v0.x * v2.y) - (v0.y * v2.x));
    Fixed w2Row = ((minY >> kSubPixelBits) * yStep2) + ((minX >> kSubPixelBits) * xStep2) + ((v1.x * v0.y) - (v1.y * v0.x));

    for (Fixed y = minY; y <= maxY; y += kSubPixelStep)
    {
        Fixed w0 = w0Row;
        Fixed w1 = w1Row;
        Fixed w2 = w2Row;

        for (Fixed x = minX; x <= maxX; x += kSubPixelStep)
        {
            const Fixed w0Biased = w0 + bias0;
            const Fixed w1Biased = w1 + bias1;
            const Fixed w2Biased = w2 + bias2;

            // If these are all positive, then the pixel lies within the triangle. We only care
            // about sign here: ORing will yield a negative value if any weights are negative.
            if ((w0Biased | w1Biased | w2Biased) >= 0)
            {
                // Barycentric interpolation. TODO: For all attributes other than depth, should
                // perspective divide here.
                const float wSum   = w0 + w1 + w2;
                const float w0Norm = static_cast<float>(w0) / wSum;
                const float w1Norm = static_cast<float>(w1) / wSum;
                const float w2Norm = static_cast<float>(w2) / wSum;

                const CVector4 colour = (w0Norm * inVertices[v0.index].colour)
                                      + (w1Norm * inVertices[v1.index].colour)
                                      + (w2Norm * inVertices[v2.index].colour);

                const Fixed pixelX = x >> kSubPixelBits;
                const Fixed pixelY = y >> kSubPixelBits;

                inSurface.WritePixel(pixelX,
                                     pixelY,
                                     colour);
            }

            w0 += xStep0;
            w1 += xStep1;
            w2 += xStep2;
        }

        w0Row += yStep0;
        w1Row += yStep1;
        w2Row += yStep2;
    }
}

#if 0
static void DrawLine(CSurface&       inSurface,
                     const uint32_t  inX0,
                     const uint32_t  inY0,
                     const uint32_t  inX1,
                     const uint32_t  inY1,
                     const CVector4& inColour)
{
    int32_t x0 = inX0;
    int32_t x1 = inX1;

    int32_t y0 = inY0;
    int32_t y1 = inY1;

    const bool transpose = std::abs(y1 - y0) > std::abs(x1 - x0);

    if (transpose)
    {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }

    if (x0 > x1)
    {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    int32_t dx = x1 - x0;
    int32_t dy = y1 - y0;

    int32_t y = y0;
    int32_t e = 0;

    for (int32_t x = x0; x <= x1; x++)
    {
        inSurface.WritePixel((transpose) ? y : x,
                             (transpose) ? x : y,
                             inColour);

        e += std::abs(dy);

        if (2 * e >= dx)
        {
            y += (dy > 0) ? 1 : -1;
            e -= dx;
        }
    }
}
#endif
