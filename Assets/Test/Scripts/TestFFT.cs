using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

//[ExecuteInEditMode]
public class TestFFT : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
    }

    public Texture2D photo;
    public ComputeShader fft;

    private FFT _fft;

    void OnDisable()
    {

    }
    void OnEnable()
    {
        _fft = new FFT(fft);
    }

    // Update is called once per frame
    void Update()
    {
        _fft.Init(photo);
        Profiler.BeginSample("DFT");
        RenderTexture result = _fft.DFT();
        Profiler.EndSample();
        GetComponent<Renderer>().sharedMaterial.mainTexture = result;
    }
}
