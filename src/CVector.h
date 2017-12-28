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

class CVector2
{
public:
                            CVector2() {}
                            explicit CVector2(const float inVal);
                            CVector2(const float inX,
                                     const float inY);

public:
    float                   x;
    float                   y;
};

inline CVector2::CVector2(const float inVal) :
    x   (inVal),
    y   (inVal)
{}

inline CVector2::CVector2(const float inX,
                          const float inY) :
    x   (inX),
    y   (inY)
{}

class CVector4
{
public:
                            CVector4() {}
                            explicit CVector4(const float inVal);
                            CVector4(const float inX,
                                     const float inY,
                                     const float inZ,
                                     const float inW);

public:
    union
    {
        float               x;
        float               r;
    };

    union
    {
        float               y;
        float               g;
    };

    union
    {
        float               z;
        float               b;
    };

    union
    {
        float               w;
        float               a;
    };
};

inline CVector4::CVector4(const float inVal) :
    x   (inVal),
    y   (inVal),
    z   (inVal),
    w   (inVal)
{}

inline CVector4::CVector4(const float inX,
                          const float inY,
                          const float inZ,
                          const float inW) :
    x   (inX),
    y   (inY),
    z   (inZ),
    w   (inW)
{}

inline CVector4 operator*(const CVector4& inLHS, const float inRHS)
{
    return CVector4(inLHS.x * inRHS,
                    inLHS.y * inRHS,
                    inLHS.z * inRHS,
                    inLHS.w * inRHS);
}

inline CVector4 operator*(const float inLHS, const CVector4& inRHS)
{
    return CVector4(inLHS * inRHS.x,
                    inLHS * inRHS.y,
                    inLHS * inRHS.z,
                    inLHS * inRHS.w);
}

inline CVector4 operator+(const CVector4& inLHS, const CVector4& inRHS)
{
    return CVector4(inLHS.x + inRHS.x,
                    inLHS.y + inRHS.y,
                    inLHS.z + inRHS.z,
                    inLHS.w + inRHS.w);
}
