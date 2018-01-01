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

#include "Types.h"

#include <smmintrin.h>

class CSIMDFloat;

// Class implementing a 32-bit signed integer with a different value for each of 4 SIMD lanes.
class CSIMDInt32
{
public:
                        CSIMDInt32() {}
                        CSIMDInt32(const __m128i inValue) : mValue(inValue) {}

                        CSIMDInt32(const int32_t inValue);

                        CSIMDInt32(const int32_t inValue0,
                                   const int32_t inValue1,
                                   const int32_t inValue2,
                                   const int32_t inValue3);

                        // Convert float to int32.
                        CSIMDInt32(const CSIMDFloat inValue);

    __m128i             GetValue() const { return mValue; }

    // Extract the value from a lane.
    static int32_t      ExtractScalar(const CSIMDInt32 inValue, const uint8_t inLane);

    // Comparison operators. Returns a mask with 0xffffffff in a lane if the comparison was true
    // for that lane, or 0 if not.
    static CSIMDInt32   LessThan(const CSIMDInt32 inLHS, const CSIMDInt32 inRHS);

    // Test whether every bit across all lanes is 1.
    static bool         AllOnes(const CSIMDInt32 inValue);

    CSIMDInt32&         operator+=(const CSIMDInt32 inRHS);

private:
    __m128i             mValue;
};

// Class implementing a single precision float with a different value for each of 4 SIMD lanes.
class CSIMDFloat
{
public:
                        CSIMDFloat() {}
                        CSIMDFloat(const __m128 inValue) : mValue(inValue) {}

                        CSIMDFloat(const float inValue);

                        CSIMDFloat(const float inValue0,
                                   const float inValue1,
                                   const float inValue2,
                                   const float inValue3);

                        // Convert int32 to float.
                        CSIMDFloat(const CSIMDInt32 inValue);

                        // Initialise from the values of a vector (X to lane 0, W to lane 3).
                        CSIMDFloat(const CVector4& inValue);

    __m128              GetValue() const { return mValue; }

    // Extract the value from a lane and return another CSIMDFloat with that value in every lane.
    static CSIMDFloat   Extract(const CSIMDFloat inValue, const uint8_t inLane);

private:
    __m128              mValue;
};

inline CSIMDInt32::CSIMDInt32(const int32_t inValue) :
    mValue  (_mm_set1_epi32(inValue))
{}

inline CSIMDInt32::CSIMDInt32(const int32_t inValue0,
                              const int32_t inValue1,
                              const int32_t inValue2,
                              const int32_t inValue3) :
    mValue  (_mm_set_epi32(inValue3,
                           inValue2,
                           inValue1,
                           inValue0))
{}

inline CSIMDInt32::CSIMDInt32(const CSIMDFloat inValue) :
    mValue  (_mm_cvtps_epi32(inValue.GetValue()))
{}

inline int32_t CSIMDInt32::ExtractScalar(const CSIMDInt32 inValue, const uint8_t inLane)
{
    // _mm_extract_epi32 requires an immediate value, it will not compile if we just pass inLane
    // directly. With any luck the compiler should constant fold this.
    switch (inLane)
    {
        case 0:  return _mm_extract_epi32(inValue.GetValue(), 0);
        case 1:  return _mm_extract_epi32(inValue.GetValue(), 1);
        case 2:  return _mm_extract_epi32(inValue.GetValue(), 2);
        case 3:  return _mm_extract_epi32(inValue.GetValue(), 3);
        default: __builtin_unreachable();
    }
}

inline CSIMDInt32 CSIMDInt32::LessThan(const CSIMDInt32 inLHS, const CSIMDInt32 inRHS)
{
    return _mm_cmplt_epi32(inLHS.GetValue(), inRHS.GetValue());
}

inline bool CSIMDInt32::AllOnes(const CSIMDInt32 inValue)
{
    return _mm_test_all_ones(inValue.GetValue());
}

inline CSIMDInt32 operator<<(const CSIMDInt32 inValue, const int32_t inShift)
{
    return _mm_slli_epi32(inValue.GetValue(), inShift);
}

inline CSIMDInt32 operator+(const CSIMDInt32 inLHS, const CSIMDInt32 inRHS)
{
    return _mm_add_epi32(inLHS.GetValue(), inRHS.GetValue());
}

inline CSIMDInt32 operator*(const CSIMDInt32 inLHS, const CSIMDInt32 inRHS)
{
    return _mm_mullo_epi32(inLHS.GetValue(), inRHS.GetValue());
}

inline CSIMDInt32 operator|(const CSIMDInt32 inLHS, const CSIMDInt32 inRHS)
{
    return _mm_or_si128(inLHS.GetValue(), inRHS.GetValue());
}

inline CSIMDInt32& CSIMDInt32::operator+=(const CSIMDInt32 inRHS)
{
    *this = *this + inRHS;
    return *this;
}

inline CSIMDFloat::CSIMDFloat(const float inValue) :
    mValue  (_mm_set1_ps(inValue))
{}

inline CSIMDFloat::CSIMDFloat(const float inValue0,
                              const float inValue1,
                              const float inValue2,
                              const float inValue3) :
    mValue  (_mm_set_ps(inValue3,
                        inValue2,
                        inValue1,
                        inValue0))
{}

inline CSIMDFloat::CSIMDFloat(const CSIMDInt32 inValue) :
    mValue  (_mm_cvtepi32_ps(inValue.GetValue()))
{}

inline CSIMDFloat::CSIMDFloat(const CVector4& inValue) :
    mValue  (_mm_load_ps(inValue.values))
{}

inline CSIMDFloat CSIMDFloat::Extract(const CSIMDFloat inValue, const uint8_t inLane)
{
    // _mm_shuffle_ps requires an immediate value, it will not compile if we just pass inLane
    // directly. With any luck the compiler should constant fold this.
    switch (inLane)
    {
        case 0:  return _mm_shuffle_ps(inValue.GetValue(), inValue.GetValue(), _MM_SHUFFLE(0, 0, 0, 0));
        case 1:  return _mm_shuffle_ps(inValue.GetValue(), inValue.GetValue(), _MM_SHUFFLE(1, 1, 1, 1));
        case 2:  return _mm_shuffle_ps(inValue.GetValue(), inValue.GetValue(), _MM_SHUFFLE(2, 2, 2, 2));
        case 3:  return _mm_shuffle_ps(inValue.GetValue(), inValue.GetValue(), _MM_SHUFFLE(3, 3, 3, 3));
        default: __builtin_unreachable();
    }
}

inline CSIMDFloat operator+(const CSIMDFloat inLHS, const CSIMDFloat inRHS)
{
    return _mm_add_ps(inLHS.GetValue(), inRHS.GetValue());
}

inline CSIMDFloat operator*(const CSIMDFloat inLHS, const CSIMDFloat inRHS)
{
    return _mm_mul_ps(inLHS.GetValue(), inRHS.GetValue());
}

inline CSIMDFloat operator/(const CSIMDFloat inLHS, const CSIMDFloat inRHS)
{
    return _mm_div_ps(inLHS.GetValue(), inRHS.GetValue());
}
