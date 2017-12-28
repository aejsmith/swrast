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

#include "CSurface.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <SDL.h>

static const uint32_t kWindowWidth  = 512;
static const uint32_t kWindowHeight = 512;

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

// TODO: Replace this with a proper vertex attribute system.
struct SVertex
{
    CVector4        position;
    CVector4        colour;
};

// Structure describing a triangle during rasterisation.
struct STriVertex
{
    // Integer position.
    int32_t         x;
    int32_t         y;

    // Index in the original vertex data.
    size_t          index;
};

// Winding of a triangle.
enum ETriWinding
{
    kTriWinding_CW,
    kTriWinding_CCW,
    kTriWinding_Degenerate,
};

static void DrawTriangle(CSurface&      inSurface,
                         const SVertex* inVertices)
{
    // Number of bits of sub-pixel precision. Snapping vertex positions to integers can cause
    // noticable artifacts, particularly with animation. Therefore, we snap to sub-pixel positions
    // instead. We use fixed-point arithmetic internally. Currently using 28.4, which is good for
    // vertex positions ranging from [-2048, 2047].
    static constexpr int32_t kSubPixelBits = 4;

    // Sub-pixel step.
    static constexpr int32_t kSubPixelStep = 1 << kSubPixelBits;

    // Sub-pixel mask.
    static constexpr int32_t kSubPixelMask = kSubPixelStep - 1;

    // We're following D3D10/GL conventions and have pixel centres at .5 offsets, i.e. position
    // (0.5, 0.5) corresponds exactly to the top left pixel.
    static constexpr float kPixelCenter = 0.5f;

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

            // To deal with the pixel center, just bias the positions to have the center at exact
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
    const int32_t bias0 = (IsTopLeft(v1, v2)) ? 0 : -1;
    const int32_t bias1 = (IsTopLeft(v2, v0)) ? 0 : -1;
    const int32_t bias2 = (IsTopLeft(v0, v1)) ? 0 : -1;

    // Calculate the bounding box of the triangle, rounded to whole pixels (min rounds down, max
    // rounds up).
    int32_t minX = std::min(v0.x, std::min(v1.x, v2.x)) & ~kSubPixelMask;
    int32_t minY = std::min(v0.y, std::min(v1.y, v2.y)) & ~kSubPixelMask;
    int32_t maxX = (std::max(v0.x, std::max(v1.x, v2.x)) + kSubPixelMask) & ~kSubPixelMask;
    int32_t maxY = (std::max(v0.y, std::max(v1.y, v2.y)) + kSubPixelMask) & ~kSubPixelMask;

    // Clip to the surface area.
    minX = std::max(minX, 0);
    minY = std::max(minY, 0);
    maxX = std::min(maxX, (static_cast<int32_t>(inSurface.GetWidth()) - 1) << kSubPixelBits);
    maxY = std::min(maxY, (static_cast<int32_t>(inSurface.GetHeight()) - 1) << kSubPixelBits);

    for (int32_t y = minY; y <= maxY; y += kSubPixelStep)
    {
        for (int32_t x = minX; x <= maxX; x += kSubPixelStep)
        {
            // Calculate barycentric coordinates of this pixel. Things are swapped around a bit
            // here because Y is down in screen coordinates.
            const int32_t w0 = ((v1.x - v2.x) * (y - v1.y)) - ((v1.y - v2.y) * (x - v1.x));
            const int32_t w1 = ((v2.x - v0.x) * (y - v2.y)) - ((v2.y - v0.y) * (x - v2.x));
            const int32_t w2 = ((v0.x - v1.x) * (y - v0.y)) - ((v0.y - v1.y) * (x - v0.x));

            // If these are all positive, then the pixel lies within the triangle.
            if (w0 + bias0 >= 0 && w1 + bias1 >= 0 && w2 + bias2 >= 0)
            {
                // Barycentric interpolation.
                // TODO: The GL spec gives different formulae for interpolating depth and other
                // attributes, what's the difference?
                const float wSum   = w0 + w1 + w2;
                const float w0Norm = static_cast<float>(w0) / wSum;
                const float w1Norm = static_cast<float>(w1) / wSum;
                const float w2Norm = static_cast<float>(w2) / wSum;

                const CVector4 colour = (w0Norm * inVertices[v0.index].colour)
                                      + (w1Norm * inVertices[v1.index].colour)
                                      + (w2Norm * inVertices[v2.index].colour);

                const int32_t pixelX = x >> kSubPixelBits;
                const int32_t pixelY = y >> kSubPixelBits;

                inSurface.WritePixel(pixelX,
                                     pixelY,
                                     colour);
            }
        }
    }
}

static void Draw(CSurface& inSurface)
{
    inSurface.Clear();

    const SVertex vertices[3] =
    {
        {{-0.5f, -0.5f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.5f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    };

    DrawTriangle(inSurface, vertices);
}

int main(int argc, char** argv)
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("swrast",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          kWindowWidth,
                                          kWindowHeight,
                                          0);

    CSurface surface(kWindowWidth, kWindowHeight);

    SDL_Surface* sourceSDLSurface = surface.CreateSDLSurface();

    SDL_Surface* destSDLSurface = SDL_GetWindowSurface(window);
    SDL_SetSurfaceBlendMode(destSDLSurface, SDL_BLENDMODE_NONE);

    double totalDrawTime  = 0.0;
    double totalFrameTime = 0.0;
    uint32_t numFrames    = 0;

    const uint64_t perfFreq = SDL_GetPerformanceFrequency();

    while (true)
    {
        const uint64_t frameStart = SDL_GetPerformanceCounter();

        SDL_Event event;
        if (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                break;
        }

        const uint64_t drawStart = SDL_GetPerformanceCounter();

        Draw(surface);

        const uint64_t drawEnd = SDL_GetPerformanceCounter();

        SDL_BlitSurface(sourceSDLSurface, nullptr, destSDLSurface, nullptr);
        SDL_UpdateWindowSurface(window);

        const uint64_t frameEnd = SDL_GetPerformanceCounter();

        totalDrawTime  += static_cast<double>((drawEnd  - drawStart)  * 1000) / perfFreq;
        totalFrameTime += static_cast<double>((frameEnd - frameStart) * 1000) / perfFreq;
        numFrames++;

        if (totalFrameTime >= 500.0)
        {
            const double avgDrawTime  = totalDrawTime  / numFrames;
            const double avgFrameTime = totalFrameTime / numFrames;

            printf("Average: draw = %.4f ms, frame = %.4f ms\n",
                   avgDrawTime,
                   avgFrameTime);

            totalDrawTime = totalFrameTime = 0.0;
            numFrames = 0;
        }
    }

    return 0;
}
