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
#include <smmintrin.h>

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
    WritePixel(inX, inY, _mm_load_ps(inColour.values));
}

void CSurface::WritePixel(const uint32_t  inX,
                          const uint32_t  inY,
                          const __m128    inColour)
{
    __m128 f;
    __m128i i;

    // Convert to integer value, with 1.0 = 255. Default rounding mode for the conversion should be
    // round to nearest.
    f = _mm_mul_ps(inColour, _mm_set1_ps(255.0f));
    i = _mm_cvtps_epi32(f);

    // Pack to signed 16-bit integers, then from that to unsigned 8-bit. This clamps to between
    // 0 and 255.
    i = _mm_packus_epi32(i, i);
    i = _mm_packus_epi16(i, i);

    // Grab the low 32 bits, which now have each value in the correct bit positions.
    // A = 24-31, B = 16-23, G = 8-15, R = 0-7.
    uint32_t pixel = _mm_cvtsi128_si32(i);

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
