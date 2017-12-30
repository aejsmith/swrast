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
#include <cstdio>
#include <SDL.h>

static const uint32_t kWindowWidth  = 512;
static const uint32_t kWindowHeight = 512;

static void Draw(CSurface&    inSurface,
                 CRasteriser& inRasteriser)
{
    const SVertex vertices[3] =
    {
        {{-0.5f, -0.5f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.5f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    };

    inRasteriser.DrawTriangle(inSurface, vertices);
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

    CRasteriser rasteriser;

    while (true)
    {
        const uint64_t frameStart = SDL_GetPerformanceCounter();

        SDL_Event event;
        if (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                break;
        }

        surface.Clear();

        const uint64_t drawStart = SDL_GetPerformanceCounter();

        Draw(surface, rasteriser);

        const uint64_t drawEnd = SDL_GetPerformanceCounter();

        SDL_BlitSurface(sourceSDLSurface, nullptr, destSDLSurface, nullptr);
        SDL_UpdateWindowSurface(window);

        const uint64_t frameEnd = SDL_GetPerformanceCounter();

        totalDrawTime  += static_cast<double>((drawEnd  - drawStart)  * 1000) / perfFreq;
        totalFrameTime += static_cast<double>((frameEnd - frameStart) * 1000) / perfFreq;
        numFrames++;

        if (totalFrameTime >= 2000.0)
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
