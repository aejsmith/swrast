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
#include "SIMDTypes.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// Number of bits of sub-pixel precision. Snapping vertex positions to integers can cause noticable
// artifacts, particularly with animation. Therefore, we snap to sub-pixel positions instead. We
// use fixed-point arithmetic internally. Currently using 28.4, which is good for vertex positions
// ranging from [-2048, 2047].
static constexpr int32_t kSubPixelBits = 4;
static constexpr int32_t kSubPixelStep = 1 << kSubPixelBits;
static constexpr int32_t kSubPixelMask = kSubPixelStep - 1;

// Loop step per quad (we rasterize 2x2 quads at a time).
static constexpr int32_t kQuadBits = kSubPixelBits + 1;
static constexpr int32_t kQuadStep = 1 << kQuadBits;
static constexpr int32_t kQuadMask = kQuadStep - 1;

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
    const int32_t weight      = ((v0.x - v1.x) * (v2.y - v0.y)) - ((v0.y - v1.y) * (v2.x - v0.x));
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

    const CSIMDInt32 bias0 = (IsTopLeft(v1, v2)) ? 0 : -1;
    const CSIMDInt32 bias1 = (IsTopLeft(v2, v0)) ? 0 : -1;
    const CSIMDInt32 bias2 = (IsTopLeft(v0, v1)) ? 0 : -1;

    // Calculate the bounding box of the triangle, rounded to whole 2x2 pixel quads (min rounds
    // down, max rounds up). Since we want to include the maximum pixels, always round up (i.e.
    // multiples of 2 are still rounded to the next).
    int32_t startX = std::min(v0.x, std::min(v1.x, v2.x)) & ~kQuadMask;
    int32_t startY = std::min(v0.y, std::min(v1.y, v2.y)) & ~kQuadMask;
    int32_t endX   = (std::max(v0.x, std::max(v1.x, v2.x)) + kQuadStep) & ~kQuadMask;
    int32_t endY   = (std::max(v0.y, std::max(v1.y, v2.y)) + kQuadStep) & ~kQuadMask;

    // FIXME: Handle non-multiple-of-2 surface sizes. Will be needed for viewport origin as well.
    // We need to make sure we don't write to pixels outside the surface.
    assert(!(inSurface.GetWidth() % 2));
    assert(!(inSurface.GetHeight() % 2));

    // Clip to the surface area.
    startX = std::max(startX, static_cast<int32_t>(0));
    startY = std::max(startY, static_cast<int32_t>(0));
    endX   = std::min(endX,  (static_cast<int32_t>(inSurface.GetWidth()))  << kSubPixelBits);
    endY   = std::min(endY,  (static_cast<int32_t>(inSurface.GetHeight())) << kSubPixelBits);

    // At each pixel, the barycentric weights are given by:
    //
    //   w0 = (x * (v2.y - v1.y)) + (y * (v1.x - v2.x)) + ((v2.x * v1.y) - (v2.y * v1.x))
    //   w1 = (x * (v0.y - v2.y)) + (y * (v2.x - v0.x)) + ((v0.x * v2.y) - (v0.y * v2.x))
    //   w2 = (x * (v1.y - v0.y)) + (y * (v0.x - v1.x)) + ((v1.x * v0.y) - (v1.y * v0.x))
    //
    // The last part is a constant term:
    const CSIMDInt32 c0         = (v2.x * v1.y) - (v2.y * v1.x);
    const CSIMDInt32 c1         = (v0.x * v2.y) - (v0.y * v2.x);
    const CSIMDInt32 c2         = (v1.x * v0.y) - (v1.y * v0.x);

    // The increment in the weight at each pixel/quad in the X direction is given as follows:
    const CSIMDInt32 xStep0     = (v2.y - v1.y) << kSubPixelBits;
    const CSIMDInt32 xStep1     = (v0.y - v2.y) << kSubPixelBits;
    const CSIMDInt32 xStep2     = (v1.y - v0.y) << kSubPixelBits;
    const CSIMDInt32 xStep0Quad = xStep0 << 1;
    const CSIMDInt32 xStep1Quad = xStep1 << 1;
    const CSIMDInt32 xStep2Quad = xStep2 << 1;

    // The same for each pixel/quad step in the Y direction:
    const CSIMDInt32 yStep0     = (v1.x - v2.x) << kSubPixelBits;
    const CSIMDInt32 yStep1     = (v2.x - v0.x) << kSubPixelBits;
    const CSIMDInt32 yStep2     = (v0.x - v1.x) << kSubPixelBits;
    const CSIMDInt32 yStep0Quad = yStep0 << 1;
    const CSIMDInt32 yStep1Quad = yStep1 << 1;
    const CSIMDInt32 yStep2Quad = yStep2 << 1;

    // Pixel offsets and min X/Y pixels for each SIMD lane.
    // Lane 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right.
    const CSIMDInt32 xLaneOffset(0, 1, 0, 1);
    const CSIMDInt32 yLaneOffset(0, 0, 1, 1);
    const CSIMDInt32 startXLane = (startX >> kSubPixelBits) + xLaneOffset;
    const CSIMDInt32 startYLane = (startY >> kSubPixelBits) + yLaneOffset;

    // Weight at the start of the row (incremented each Y iteration).
    CSIMDInt32 w0Row = (startYLane * yStep0) + (startXLane * xStep0) + c0;
    CSIMDInt32 w1Row = (startYLane * yStep1) + (startXLane * xStep1) + c1;
    CSIMDInt32 w2Row = (startYLane * yStep2) + (startXLane * xStep2) + c2;

    const CSIMDFloat colour0 = inVertices[v0.index].colour;
    const CSIMDFloat colour1 = inVertices[v1.index].colour;
    const CSIMDFloat colour2 = inVertices[v2.index].colour;

    const CSIMDInt32 zero = 0;

    for (int32_t y = startY; y < endY; y += kQuadStep)
    {
        CSIMDInt32 w0 = w0Row;
        CSIMDInt32 w1 = w1Row;
        CSIMDInt32 w2 = w2Row;

        for (int32_t x = startX; x < endX; x += kQuadStep)
        {
            // Sets each lane to 0xffffffff if the weight is < 0, i.e. the pixel is outside the
            // triangle.
            const CSIMDInt32 cmp0 = CSIMDInt32::LessThan(w0 + bias0, zero);
            const CSIMDInt32 cmp1 = CSIMDInt32::LessThan(w1 + bias1, zero);
            const CSIMDInt32 cmp2 = CSIMDInt32::LessThan(w2 + bias2, zero);

            // OR all of them together: if the result is not all ones, then at least one of the
            // pixels is inside the triangle (all weights >= 0 for a lane).
            const CSIMDInt32 mask = cmp0 | cmp1 | cmp2;
            if (!CSIMDInt32::AllOnes(mask))
            {
                // Barycentric interpolation. Calculate the normalised barycentric weights.
                const CSIMDFloat wSum   = static_cast<CSIMDFloat>(w0 + w1 + w2);
                const CSIMDFloat w0Norm = static_cast<CSIMDFloat>(w0) / wSum;
                const CSIMDFloat w1Norm = static_cast<CSIMDFloat>(w1) / wSum;
                const CSIMDFloat w2Norm = static_cast<CSIMDFloat>(w2) / wSum;

                // TODO: Can this be vectorised better?
                for (uint8_t lane = 0; lane < 4; lane++)
                {
                    if (CSIMDInt32::ExtractScalar(mask, lane) == 0)
                    {
                        // This pixel is inside the triangle (see above).
                        const int32_t pixelX = (x >> kSubPixelBits) + (lane & 1);
                        const int32_t pixelY = (y >> kSubPixelBits) + (lane >> 1);

                        // Extract weights for this pixel.
                        const CSIMDFloat pixelW0 = CSIMDFloat::Extract(w0Norm, lane);
                        const CSIMDFloat pixelW1 = CSIMDFloat::Extract(w1Norm, lane);
                        const CSIMDFloat pixelW2 = CSIMDFloat::Extract(w2Norm, lane);

                        // TODO: For all attributes other than depth, should perspective divide
                        // here.
                        const CSIMDFloat colour = pixelW0 * colour0
                                                + pixelW1 * colour1
                                                + pixelW2 * colour2;

                        inSurface.WritePixel(pixelX,
                                             pixelY,
                                             colour.GetValue());
                    }
                }
            }

            w0 += xStep0Quad;
            w1 += xStep1Quad;
            w2 += xStep2Quad;
        }

        w0Row += yStep0Quad;
        w1Row += yStep1Quad;
        w2Row += yStep2Quad;
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
