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

#include <SDL.h>

CSurface::CSurface(const uint32_t inWidth,
                   const uint32_t inHeight) :
    mWidth  (inWidth),
    mHeight (inHeight)
{
    const size_t dataSize = sizeof(uint32_t) * inWidth * inHeight;
    mData.reset(new uint8_t[dataSize]);
}

void CSurface::Clear()
{
    memset(mData.get(), 0, sizeof(uint32_t) * mWidth * mHeight);
}

void CSurface::WritePixel(const uint32_t  inX,
                          const uint32_t  inY,
                          const CVector4& inColour)
{
    uint32_t pixel = 0;

    pixel |= (lround(inColour.a * 255) & 0xff) << 24;
    pixel |= (lround(inColour.b * 255) & 0xff) << 16;
    pixel |= (lround(inColour.g * 255) & 0xff) << 8;
    pixel |= (lround(inColour.r * 255) & 0xff) << 0;

    const size_t offset = ((inY * mHeight) + inX) * sizeof(uint32_t);

    *reinterpret_cast<uint32_t*>(&mData[offset]) = pixel;
}

SDL_Surface* CSurface::CreateSDLSurface()
{
    return SDL_CreateRGBSurfaceFrom(mData.get(),
                                    mWidth,
                                    mHeight,
                                    32,
                                    sizeof(uint32_t) * mWidth,
                                    0x000000ff,
                                    0x0000ff00,
                                    0x00ff0000,
                                    0xff000000);
}
