using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Rendering;
using System.Runtime.InteropServices;
using System;

public enum FFTType
{
    DFT = 0,
    IDFT = 1
}

public enum NormalizeType
{
    DFT,             // dft will be normalized by 1.0f / n
    IDFT,            // idft will be normalized by 1.0f / n
    SYMMETRIC,       // dft and idft will be normalized by 1.0f / sqrt(n) both.
}

public class FFT : System.IDisposable
{
    // temp texture for two pass
    private RenderTexture fftTextureTmp_;
    // output texture
    private RenderTexture fftTextureOut_;

    private Texture fftTextureIn_;

    private ComputeShader fftShader_;

    private const int KERNEL_SCALE = 0;
    private const int KERNEL_DIT_X = 1;
    private const int KERNEL_DIT_Y = 2;
    private const int KERNEL_DIT_X_IDFT = 3;
    private const int KERNEL_DIT_Y_IDFT = 4;

    private const int KERNEL_DIT4_X = 5;
    private const int KERNEL_DIT4_Y = 6;
    private const int KERNEL_DIT4_X_IDFT = 7;
    private const int KERNEL_DIT4_Y_IDFT = 8;

    private const int KERNEL_DIT8_X = 9;
    private const int KERNEL_DIT8_Y = 10;
    private const int KERNEL_DIT8_X_IDFT = 11;
    private const int KERNEL_DIT8_Y_IDFT = 12;

    private const int KERNEL_DIT16_X = 13;
    private const int KERNEL_DIT16_Y = 14;
    private const int KERNEL_DIT16_X_IDFT = 15;
    private const int KERNEL_DIT16_Y_IDFT = 16;

    private const int KERNEL_DIT32_X = 17;
    private const int KERNEL_DIT32_Y = 18;
    private const int KERNEL_DIT32_X_IDFT = 19;
    private const int KERNEL_DIT32_Y_IDFT = 20;

    private const int KERNEL_DIT64_X = 21;
    private const int KERNEL_DIT64_Y = 22;
    private const int KERNEL_DIT64_X_IDFT = 23;
    private const int KERNEL_DIT64_Y_IDFT = 24;

    private const int KERNEL_DIT128_X = 25;
    private const int KERNEL_DIT128_Y = 26;
    private const int KERNEL_DIT128_X_IDFT = 27;
    private const int KERNEL_DIT128_Y_IDFT = 28;

    private const int KERNEL_DIT256_X = 29;
    private const int KERNEL_DIT256_Y = 30;
    private const int KERNEL_DIT256_X_IDFT = 31;
    private const int KERNEL_DIT256_Y_IDFT = 32;

    private const int KERNEL_DIT512_X = 33;
    private const int KERNEL_DIT512_Y = 34;
    private const int KERNEL_DIT512_X_IDFT = 35;
    private const int KERNEL_DIT512_Y_IDFT = 36;

    private const int KERNEL_DIT1024_X = 37;
    private const int KERNEL_DIT1024_Y = 38;
    private const int KERNEL_DIT1024_X_IDFT = 39;
    private const int KERNEL_DIT1024_Y_IDFT = 40;

    private const string SHADER_NAME_N = "n";
    private const string SHADER_NAME_P = "p";

    private const string SHADER_NAME_FFT_IN = "fftin";
    private const string SHADER_NAME_FFT_OUT = "fftout";
    private const string SHADER_NAME_SCALE = "scale";

    private const int THREAD_SIZE = 1;

    private const float pi = 3.14159265359f;

    private int n;
    private int groupN;
    private int halfGroupN;

    public FFT(ComputeShader fft)
    {
        fftShader_ = fft;
    }

    public void Dispose()
    {
        if (fftTextureTmp_ != null)
            fftTextureTmp_.Release();
        if (fftTextureOut_ != null)
            fftTextureOut_.Release();
    }

    private bool IsPOT(int x)
    {
        return x >= 0 && ((x & (x - 1)) == 0);
    }

    public bool Init(Texture fftTextureIn) {
        n = fftTextureIn.width;
        groupN = n / THREAD_SIZE;
        halfGroupN = n / (THREAD_SIZE * 2);
        if (!IsPOT(n)) return false;

        fftShader_.SetInt(SHADER_NAME_N, n);

        if (fftTextureTmp_ != null)
            fftTextureTmp_.Release();
        if (fftTextureOut_ != null)
            fftTextureOut_.Release();
        fftTextureTmp_ = AllocateTexture(fftTextureIn);
        fftTextureOut_ = AllocateTexture(fftTextureIn);
        fftTextureIn_ = fftTextureIn;
        return true;
    }

    private RenderTexture AllocateTexture(Texture fftTextureIn) {
        RenderTexture tex = new RenderTexture(n, n, 0, RenderTextureFormat.RGFloat, RenderTextureReadWrite.Linear);
        tex.filterMode = fftTextureIn.filterMode;
        tex.wrapMode = fftTextureIn.wrapMode;
        tex.enableRandomWrite = true;
        tex.Create();
        return tex;
    }

    private void SwapTex()
    {
        var tempTex = fftTextureTmp_;
        fftTextureTmp_ = fftTextureOut_;
        fftTextureOut_ = tempTex;
    }

    public RenderTexture DFT(NormalizeType normalizeType=NormalizeType.DFT)
    {
        FFTUWithRadix(FFTType.DFT, 32);
        FFTVWithRadix(FFTType.DFT, 32);
        
        if (normalizeType == NormalizeType.DFT)
        {
            Normalize(1.0f / n);
        }
        else if (normalizeType == NormalizeType.SYMMETRIC)
        {
            Normalize(1.0f / Mathf.Sqrt(n));
        }
        else
        {
            SwapTex();
        }

        return fftTextureOut_;
    }

    public RenderTexture IDFT(NormalizeType normalizeType=NormalizeType.IDFT)
    {
        FFTUWithRadix(FFTType.IDFT, 32);
        FFTVWithRadix(FFTType.IDFT, 32);
        if (normalizeType == NormalizeType.IDFT)
        {
            Normalize(1.0f / n);
        }
        else if (normalizeType == NormalizeType.SYMMETRIC)
        {
            Normalize(1.0f / Mathf.Sqrt(n));
        }
        else
        {
            SwapTex();
        }

        return fftTextureOut_;
    }

    public void FFTU(FFTType fftType)
    {
        int ditXKernel = fftType == FFTType.DFT ? KERNEL_DIT_X : KERNEL_DIT_X_IDFT;

        uint p = 1;
        if (p < n)
        {
            fftShader_.SetTexture(ditXKernel, SHADER_NAME_FFT_IN, fftTextureIn_);
            fftShader_.SetTexture(ditXKernel, SHADER_NAME_FFT_OUT, fftTextureOut_);
            fftShader_.SetInt(SHADER_NAME_P, (int)p);

            fftTextureOut_.DiscardContents();
            fftShader_.Dispatch(ditXKernel, halfGroupN, groupN, 1);
            SwapTex();
            p <<= 1;

            while (p < n)
            {
                fftShader_.SetTexture(ditXKernel, SHADER_NAME_FFT_IN, fftTextureTmp_);
                fftShader_.SetTexture(ditXKernel, SHADER_NAME_FFT_OUT, fftTextureOut_);
                fftShader_.SetInt(SHADER_NAME_P, (int)p);

                fftTextureOut_.DiscardContents();
                fftShader_.Dispatch(ditXKernel, halfGroupN, groupN, 1);
                SwapTex();
                p <<= 1;
                
            }
        }
    }

    public void FFTV(FFTType fftType)
    {
        int ditYKernel = fftType == FFTType.DFT ? KERNEL_DIT_Y : KERNEL_DIT_Y_IDFT;

        uint p = 1;
        while (p < n)
        {
            fftShader_.SetTexture(ditYKernel, SHADER_NAME_FFT_IN, fftTextureTmp_);
            fftShader_.SetTexture(ditYKernel, SHADER_NAME_FFT_OUT, fftTextureOut_);
            fftShader_.SetInt(SHADER_NAME_P, (int)p);

            fftTextureOut_.DiscardContents();
            fftShader_.Dispatch(ditYKernel, groupN, halfGroupN, 1);
            SwapTex();
            p <<= 1;
        }
    }

    public void Normalize(float scale)
    {
        fftShader_.SetTexture(KERNEL_SCALE, SHADER_NAME_FFT_IN, fftTextureTmp_);
        fftShader_.SetTexture(KERNEL_SCALE, SHADER_NAME_FFT_OUT, fftTextureOut_);
        fftShader_.SetFloat(SHADER_NAME_SCALE, scale);

        fftTextureOut_.DiscardContents();
        fftShader_.Dispatch(KERNEL_SCALE, groupN, groupN, 1);
    }

    public void DispatchFFTU(FFTType fftType, int radix, int p, Texture fftTextureIn, RenderTexture fftTextureOut)
    {
        int ditKernel = 0;
        switch(radix)
        {
            case 2: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT_X : KERNEL_DIT_X_IDFT; break;
            case 4: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT4_X : KERNEL_DIT4_X_IDFT; break;
            case 8: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT8_X : KERNEL_DIT8_X_IDFT; break; 
            case 16: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT16_X : KERNEL_DIT16_X_IDFT; break; 
            case 32: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT32_X : KERNEL_DIT32_X_IDFT; break;
            case 64: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT64_X : KERNEL_DIT64_X_IDFT; break;
            case 128: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT128_X : KERNEL_DIT128_X_IDFT; break;
            case 256: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT256_X : KERNEL_DIT256_X_IDFT; break;
            case 512: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT512_X : KERNEL_DIT512_X_IDFT; break;
            case 1024: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT1024_X : KERNEL_DIT1024_X_IDFT; break;
        }
        fftShader_.SetTexture(ditKernel, SHADER_NAME_FFT_IN, fftTextureIn);
        fftShader_.SetTexture(ditKernel, SHADER_NAME_FFT_OUT, fftTextureOut);
        fftShader_.SetInt(SHADER_NAME_P, (int)p);

        fftTextureOut.DiscardContents();
        fftShader_.Dispatch(ditKernel, groupN / radix, groupN, 1);
        SwapTex();
    }

    public void DispatchFFTV(FFTType fftType, int radix, int p, Texture fftTextureIn, RenderTexture fftTextureOut)
    {
        int ditKernel = 0;
        switch (radix)
        {
            case 2: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT_Y : KERNEL_DIT_Y_IDFT; break;
            case 4: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT4_Y : KERNEL_DIT4_Y_IDFT; break;
            case 8: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT8_Y : KERNEL_DIT8_Y_IDFT; break;
            case 16: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT16_Y : KERNEL_DIT16_Y_IDFT; break;
            case 32: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT32_Y : KERNEL_DIT32_Y_IDFT; break;
            case 64: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT64_Y : KERNEL_DIT64_Y_IDFT; break;
            case 128: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT128_Y : KERNEL_DIT128_Y_IDFT; break;
            case 256: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT256_Y : KERNEL_DIT256_Y_IDFT; break;
            case 512: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT512_Y : KERNEL_DIT512_Y_IDFT; break;
            case 1024: ditKernel = fftType == FFTType.DFT ? KERNEL_DIT1024_Y : KERNEL_DIT1024_Y_IDFT; break;
        }
        fftShader_.SetTexture(ditKernel, SHADER_NAME_FFT_IN, fftTextureIn);
        fftShader_.SetTexture(ditKernel, SHADER_NAME_FFT_OUT, fftTextureOut);
        fftShader_.SetInt(SHADER_NAME_P, (int)p);

        fftTextureOut.DiscardContents();
        fftShader_.Dispatch(ditKernel, groupN, groupN / radix, 1);
        SwapTex();
    }


    public void FFTU4(FFTType fftType) {
        int p = 1;
        if (p < n) {
            if (p * 4 <= n) {
                DispatchFFTU(fftType, 4, p, fftTextureIn_, fftTextureOut_);
                p <<= 2;
            } else {
                DispatchFFTU(fftType, 2, p, fftTextureIn_, fftTextureOut_);
                p <<= 1;
            }

            while (p < n) {
                if (p * 4 <= n) {
                    DispatchFFTU(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 2;
                } else {
                    DispatchFFTU(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 1;
                }
            }
        }
    }

    public void FFTV4(FFTType fftType)
    {
        int p = 1;
        while (p < n)
        {
            if (p * 4 <= n)
            {
                DispatchFFTV(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTV(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                p <<= 1;
            }

        }
    }

    public void FFTU8(FFTType fftType)
    {
        int p = 1;
        if (p < n)
        {
            if (p * 8 <= n)
            {
                DispatchFFTU(fftType, 8, p, fftTextureIn_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTU(fftType, 4, p, fftTextureIn_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTU(fftType, 2, p, fftTextureIn_, fftTextureOut_);
                p <<= 1;
            }


            while (p < n)
            {
                if (p * 8 <= n)
                {
                    DispatchFFTU(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 3;
                }
                else if (p * 4 <= n)
                {
                    DispatchFFTU(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 2;
                }
                else
                {
                    DispatchFFTU(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 1;
                }
            }
        }
    }

    public void FFTV8(FFTType fftType)
    {
        int p = 1;
        while (p < n)
        {
            if (p * 8 <= n)
            {
                DispatchFFTV(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTV(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTV(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                p <<= 1;
            }

        }
    }

    public void FFTU16(FFTType fftType)
    {
        int p = 1;
        if (p < n)
        {
            if (p * 16 <= n)
            {
                DispatchFFTU(fftType, 16, p, fftTextureIn_, fftTextureOut_);
                p <<= 4;
            }
            else if (p * 8 <= n)
            {
                DispatchFFTU(fftType, 8, p, fftTextureIn_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTU(fftType, 4, p, fftTextureIn_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTU(fftType, 2, p, fftTextureIn_, fftTextureOut_);
                p <<= 1;
            }


            while (p < n)
            {
                if (p * 16 <= n)
                {
                    DispatchFFTU(fftType, 16, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 4;
                }
                else if (p * 8 <= n)
                {
                    DispatchFFTU(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 3;
                }
                else if (p * 4 <= n)
                {
                    DispatchFFTU(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 2;
                }
                else
                {
                    DispatchFFTU(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 1;
                }
            }
        }
    }

    public void FFTV16(FFTType fftType)
    {
        int p = 1;
        while (p < n)
        {
            if (p * 16 <= n)
            {
                DispatchFFTV(fftType, 16, p, fftTextureTmp_, fftTextureOut_);
                p <<= 4;
            }
            else if (p * 8 <= n)
            {
                DispatchFFTV(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTV(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTV(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                p <<= 1;
            }

        }
    }

    public void FFTU32(FFTType fftType)
    {
        int p = 1;
        if (p < n)
        {
            
            if (p * 32 <= n)
            {
                DispatchFFTU(fftType, 32, p, fftTextureIn_, fftTextureOut_);
                p <<= 5;
            } else
            if (p * 16 <= n)
            {
                DispatchFFTU(fftType, 16, p, fftTextureIn_, fftTextureOut_);
                p <<= 4;
            }
            else if (p * 8 <= n)
            {
                DispatchFFTU(fftType, 8, p, fftTextureIn_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTU(fftType, 4, p, fftTextureIn_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTU(fftType, 2, p, fftTextureIn_, fftTextureOut_);
                p <<= 1;
            }


            while (p < n)
            {
                if (p * 32 <= n)
                {
                    DispatchFFTU(fftType, 32, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 5;
                } else
                if (p * 16 <= n)
                {
                    DispatchFFTU(fftType, 16, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 4;
                }
                else if (p * 8 <= n)
                {
                    DispatchFFTU(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 3;
                }
                else if (p * 4 <= n)
                {
                    DispatchFFTU(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 2;
                }
                else
                {
                    DispatchFFTU(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                    p <<= 1;
                }
            }
        }
    }

    public void FFTV32(FFTType fftType)
    {
        int p = 1;
        while (p < n)
        {
            if (p * 32 <= n)
            {
                DispatchFFTV(fftType, 32, p, fftTextureTmp_, fftTextureOut_);
                p <<= 5;
            } else
            if (p * 16 <= n)
            {
                DispatchFFTV(fftType, 16, p, fftTextureTmp_, fftTextureOut_);
                p <<= 4;
            }
            else if (p * 8 <= n)
            {
                DispatchFFTV(fftType, 8, p, fftTextureTmp_, fftTextureOut_);
                p <<= 3;
            }
            else if (p * 4 <= n)
            {
                DispatchFFTV(fftType, 4, p, fftTextureTmp_, fftTextureOut_);
                p <<= 2;
            }
            else
            {
                DispatchFFTV(fftType, 2, p, fftTextureTmp_, fftTextureOut_);
                p <<= 1;
            }

        }
    }

    public void FFTUWithRadix(FFTType fftType, int maxRadix)
    {
        int maxRadixPow = Math.Max(2, (int)(Math.Log(maxRadix, 2)));
        int p = 1;
        if (p < n)
        {
            int radixSize = 1 << maxRadixPow;
            int radix = maxRadixPow;
            while(radixSize >= 2)
            {
                if (p * radixSize <= n)
                {
                    DispatchFFTU(fftType, radixSize, p, fftTextureIn_, fftTextureOut_);
                    p <<= radix;
                    break;
                }
                radixSize >>= 1;
                radix -= 1;
            }
            
            while (p < n)
            {

                radixSize = 1 << maxRadixPow;
                radix = maxRadixPow;
                while (radixSize >= 2)
                {
                    if (p * radixSize <= n)
                    {
                        DispatchFFTU(fftType, radixSize, p, fftTextureTmp_, fftTextureOut_);
                        p <<= radix;
                        break;
                    }
                    radixSize >>= 1;
                    radix -= 1;
                }
            }
        }
    }

    public void FFTVWithRadix(FFTType fftType, int maxRadix)
    {
        int maxRadixPow = Math.Max(1, (int)(Math.Log(maxRadix, 2)));
        int p = 1;
        while (p < n)
        {
            int radixSize = 1 << maxRadixPow;
            int radix = maxRadixPow;
            while (radixSize >= 2)
            {
                if (p * radixSize <= n)
                {
                    DispatchFFTV(fftType, radixSize, p, fftTextureTmp_, fftTextureOut_);
                    p <<= radix;
                    break;
                }
                radixSize >>= 1;
                radix -= 1;
            }

        }
    }
}
