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

#pragma once

#include "CVector.h"

#include <cstdint>
#include <cstddef>

class CSurface;

// Input vertex data structure.
// TODO: Replace this with a proper vertex attribute system.
struct SVertex
{
    CVector4                position;
    CVector4                colour;
};

// Class implementing the rasteriser.
class CRasteriser
{
public:
                            CRasteriser() {}
                            ~CRasteriser() {}

    void                    DrawTriangle(CSurface&      inSurface,
                                         const SVertex* inVertices);

private:
    // Structure describing a vertex during rasterisation.
    struct STriVertex
    {
        // Integer position.
        int32_t             x;
        int32_t             y;

        // Index in the original vertex data (we may swap vertices around to fix winding).
        size_t              index;
    };

    // Winding of a triangle.
    enum ETriWinding
    {
        kTriWinding_CW,
        kTriWinding_CCW,
        kTriWinding_Degenerate,
    };
};
