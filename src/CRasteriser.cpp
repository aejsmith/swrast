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
#include <cassert>
#include <cmath>
#include <smmintrin.h>

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

// _mm_shuffle_ps requires an immediate index. This mess is to work around that.
template <size_t lane>
struct ExtractLane
{
    float ExtractFloat(const __m128 inValue) const
    {
        return _mm_cvtss_f32(_mm_shuffle_ps(inValue, inValue, lane));
    }

    __m128 ExtractVector(const __m128 inValue) const
    {
        return _mm_shuffle_ps(inValue, inValue, _MM_SHUFFLE(lane, lane, lane, lane));
    }
};

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

    const __m128i bias0 = _mm_set1_epi32((IsTopLeft(v1, v2)) ? 0 : -1);
    const __m128i bias1 = _mm_set1_epi32((IsTopLeft(v2, v0)) ? 0 : -1);
    const __m128i bias2 = _mm_set1_epi32((IsTopLeft(v0, v1)) ? 0 : -1);

    // Calculate the bounding box of the triangle, rounded to whole 2x2 pixel quads (min rounds
    // down, max rounds up).
    int32_t minX = std::min(v0.x, std::min(v1.x, v2.x)) & ~kQuadMask;
    int32_t minY = std::min(v0.y, std::min(v1.y, v2.y)) & ~kQuadMask;
    int32_t maxX = (std::max(v0.x, std::max(v1.x, v2.x)) + kQuadMask) & ~kQuadMask;
    int32_t maxY = (std::max(v0.y, std::max(v1.y, v2.y)) + kQuadMask) & ~kQuadMask;

    // FIXME: Handle non-multiple-of-2 surface sizes. Will be needed for viewport origin as well.
    // We need to make sure we don't write to pixels outside the surface.
    assert(!(inSurface.GetWidth() % 2));
    assert(!(inSurface.GetHeight() % 2));

    // Clip to the surface area.
    minX = std::max(minX, static_cast<int32_t>(0));
    minY = std::max(minY, static_cast<int32_t>(0));
    maxX = std::min(maxX, (static_cast<int32_t>(inSurface.GetWidth()) - 1) << kSubPixelBits);
    maxY = std::min(maxY, (static_cast<int32_t>(inSurface.GetHeight()) - 1) << kSubPixelBits);

    // At each pixel, the barycentric weights are given by:
    //
    //   w0 = (x * (v2.y - v1.y)) + (y * (v1.x - v2.x)) + ((v2.x * v1.y) - (v2.y * v1.x))
    //   w1 = (x * (v0.y - v2.y)) + (y * (v2.x - v0.x)) + ((v0.x * v2.y) - (v0.y * v2.x))
    //   w2 = (x * (v1.y - v0.y)) + (y * (v0.x - v1.x)) + ((v1.x * v0.y) - (v1.y * v0.x))
    //
    // The last part is a constant term:
    const __m128i c0 = _mm_set1_epi32((v2.x * v1.y) - (v2.y * v1.x));
    const __m128i c1 = _mm_set1_epi32((v0.x * v2.y) - (v0.y * v2.x));
    const __m128i c2 = _mm_set1_epi32((v1.x * v0.y) - (v1.y * v0.x));

    // The increment in the weight at each pixel/quad in the X direction is given as follows:
    const __m128i xStep0     = _mm_set1_epi32((v2.y - v1.y) << kSubPixelBits);
    const __m128i xStep1     = _mm_set1_epi32((v0.y - v2.y) << kSubPixelBits);
    const __m128i xStep2     = _mm_set1_epi32((v1.y - v0.y) << kSubPixelBits);
    const __m128i xStep0Quad = _mm_slli_epi32(xStep0, 1);
    const __m128i xStep1Quad = _mm_slli_epi32(xStep1, 1);
    const __m128i xStep2Quad = _mm_slli_epi32(xStep2, 1);

    // The same for each pixel/quad step in the Y direction:
    const __m128i yStep0     = _mm_set1_epi32((v1.x - v2.x) << kSubPixelBits);
    const __m128i yStep1     = _mm_set1_epi32((v2.x - v0.x) << kSubPixelBits);
    const __m128i yStep2     = _mm_set1_epi32((v0.x - v1.x) << kSubPixelBits);
    const __m128i yStep0Quad = _mm_slli_epi32(yStep0, 1);
    const __m128i yStep1Quad = _mm_slli_epi32(yStep1, 1);
    const __m128i yStep2Quad = _mm_slli_epi32(yStep2, 1);

    // Pixel offsets and min X/Y pixels for each SIMD lane.
    // Lane 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right.
    const __m128i xLaneOffset = _mm_set_epi32(1, 0, 1, 0);
    const __m128i yLaneOffset = _mm_set_epi32(1, 1, 0, 0);
    const __m128i laneMinX    = _mm_add_epi32(_mm_set1_epi32(minX >> kSubPixelBits), xLaneOffset);
    const __m128i laneMinY    = _mm_add_epi32(_mm_set1_epi32(minY >> kSubPixelBits), yLaneOffset);

    // Weight at the start of the row (incremented each Y iteration).
    // w0Row = ((minY >> kSubPixelBits) * yStep0) + ((minX >> kSubPixelBits) * xStep0) + c0
    __m128i w0Row = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(laneMinY, yStep0), _mm_mullo_epi32(laneMinX, xStep0)), c0);
    __m128i w1Row = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(laneMinY, yStep1), _mm_mullo_epi32(laneMinX, xStep1)), c1);
    __m128i w2Row = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(laneMinY, yStep2), _mm_mullo_epi32(laneMinX, xStep2)), c2);

    const __m128 colour0 = _mm_load_ps(inVertices[v0.index].colour.values);
    const __m128 colour1 = _mm_load_ps(inVertices[v1.index].colour.values);
    const __m128 colour2 = _mm_load_ps(inVertices[v2.index].colour.values);

    for (int32_t y = minY; y <= maxY; y += kQuadStep)
    {
        __m128i w0 = w0Row;
        __m128i w1 = w1Row;
        __m128i w2 = w2Row;

        for (int32_t x = minX; x <= maxX; x += kQuadStep)
        {
            const __m128i zero = _mm_set1_epi32(0);

            // Sets each lane to 0xffffffff if the weight is < 0, i.e. the pixel is outside the
            // triangle.
            const __m128i cmp0 = _mm_cmplt_epi32(_mm_add_epi32(w0, bias0), zero);
            const __m128i cmp1 = _mm_cmplt_epi32(_mm_add_epi32(w1, bias1), zero);
            const __m128i cmp2 = _mm_cmplt_epi32(_mm_add_epi32(w2, bias2), zero);

            // OR all of them together: if the result is not all ones, then at least one of the
            // pixels is inside the triangle (all weights >= 0 for a lane).
            const __m128i mask = _mm_or_si128(_mm_or_si128(cmp0, cmp1), cmp2);
            if (!_mm_test_all_ones(mask))
            {
                // Barycentric interpolation. Calculate the normalised barycentric weights.
                const __m128 wSum   = _mm_cvtepi32_ps(_mm_add_epi32(_mm_add_epi32(w0, w1), w2));
                const __m128 w0Norm = _mm_div_ps(_mm_cvtepi32_ps(w0), wSum);
                const __m128 w1Norm = _mm_div_ps(_mm_cvtepi32_ps(w1), wSum);
                const __m128 w2Norm = _mm_div_ps(_mm_cvtepi32_ps(w2), wSum);

                // More mess to work around _mm_shuffle_ps requiring a constant.
                // TODO: Can this be vectorised better?
                auto DoPixel =
                    [&] (const uint8_t inLane, const auto& inExtractor)
                    {
                        if (_mm_extract_epi32(mask, inLane) == 0)
                        {
                            // This pixel is inside the triangle (see above).
                            const int32_t pixelX = (x >> kSubPixelBits) + (inLane & 1);
                            const int32_t pixelY = (y >> kSubPixelBits) + (inLane >> 1);

                            // Extract weights for this pixel.
                            const __m128 pixelW0 = inExtractor.ExtractVector(w0Norm);
                            const __m128 pixelW1 = inExtractor.ExtractVector(w1Norm);
                            const __m128 pixelW2 = inExtractor.ExtractVector(w2Norm);

                            // TODO: For all attributes other than depth, should perspective divide
                            // here.
                            CVector4 colour;
                            _mm_store_ps(colour.values,
                                         _mm_add_ps(_mm_add_ps(_mm_mul_ps(pixelW0, colour0),
                                                                          _mm_mul_ps(pixelW1, colour1)),
                                                                          _mm_mul_ps(pixelW2, colour2)));

                            inSurface.WritePixel(pixelX,
                                                 pixelY,
                                                 colour);
                        }
                    };

                DoPixel(0, ExtractLane<0>());
                DoPixel(1, ExtractLane<1>());
                DoPixel(2, ExtractLane<2>());
                DoPixel(3, ExtractLane<3>());
            }

            w0 = _mm_add_epi32(w0, xStep0Quad);
            w1 = _mm_add_epi32(w1, xStep1Quad);
            w2 = _mm_add_epi32(w2, xStep2Quad);
        }

        w0Row = _mm_add_epi32(w0Row, yStep0Quad);
        w1Row = _mm_add_epi32(w1Row, yStep1Quad);
        w2Row = _mm_add_epi32(w2Row, yStep2Quad);
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
