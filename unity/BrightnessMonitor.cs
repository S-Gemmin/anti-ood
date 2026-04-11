// ═══════════════════════════════════════════════════════════════════════════════
//  BrightnessMonitor.cs
//  Samples the agent's camera at a fixed rate and exposes current brightness
//  b_t and its temporal derivative ḃ_t to downstream OOD controllers.
// ═══════════════════════════════════════════════════════════════════════════════

using UnityEngine;

namespace OOD.Perception
{
    /// <summary>
    /// Captures average luminance from a <see cref="Camera"/> render texture each
    /// sampling tick and maintains a smoothed first-order derivative.
    /// <para>
    ///   Both <see cref="Brightness"/> (<c>b_t</c>) and
    ///   <see cref="BrightnessRate"/> (<c>ḃ_t</c>) are available as properties
    ///   for use by any OOD controller.
    /// </para>
    /// </summary>
    [AddComponentMenu("OOD / Perception / Brightness Monitor")]
    public sealed class BrightnessMonitor : MonoBehaviour
    {

        [Header("Camera Source")]
        [Tooltip("Camera whose view is sampled for brightness. Defaults to main camera.")]
        public Camera targetCamera;

        [Header("Sampling")]
        [Tooltip("Resolution of the thumbnail used for brightness estimation. " +
                 "Smaller = faster; larger = more accurate.")]
        public int thumbnailResolution = 32;

        [Tooltip("How many times per second to sample brightness.")]
        [Range(1, 60)] public int samplesPerSecond = 10;

        [Header("Derivative Smoothing")]
        [Tooltip("Exponential smoothing factor for the brightness derivative. " +
                 "0 = no smoothing; 1 = infinitely smooth.")]
        [Range(0f, 0.99f)] public float derivativeSmoothing = 0.8f;


        /// <summary>Current normalised scene brightness b_t ∈ [0, 1].</summary>
        public float Brightness { get; private set; } = 1f;

        /// <summary>Smoothed rate of change ḃ_t (units per second). Negative = darkening.</summary>
        public float BrightnessRate { get; private set; } = 0f;


        private RenderTexture _rt;
        private Texture2D _thumb;
        private float _sampleInterval;
        private float _nextSampleTime;
        private float _previousBrightness;


        private void Awake()
        {
            if (targetCamera == null)
                targetCamera = Camera.main;

            _sampleInterval = 1f / samplesPerSecond;
            _previousBrightness = 1f;

            // Allocate a tiny render texture for cheap GPU readback
            _rt = new RenderTexture(thumbnailResolution, thumbnailResolution, 0,
                                   RenderTextureFormat.ARGB32);
            _thumb = new Texture2D(thumbnailResolution, thumbnailResolution,
                                   TextureFormat.RGB24, false);

            Debug.Log($"[BrightnessMonitor] Sampling at {samplesPerSecond} Hz " +
                      $"using {thumbnailResolution}² thumbnail.");
        }

        private void Update()
        {
            if (Time.time < _nextSampleTime) return;
            _nextSampleTime = Time.time + _sampleInterval;

            SampleBrightness();
        }

        private void OnDestroy()
        {
            if (_rt != null) _rt.Release();
            if (_rt != null) Destroy(_rt);
            if (_thumb != null) Destroy(_thumb);
        }


        private void SampleBrightness()
        {
            // Render current camera view into thumbnail RT
            var previousTarget = targetCamera.targetTexture;
            targetCamera.targetTexture = _rt;
            targetCamera.Render();
            targetCamera.targetTexture = previousTarget;

            // GPU readback
            var previousRT = RenderTexture.active;
            RenderTexture.active = _rt;
            _thumb.ReadPixels(new Rect(0, 0, thumbnailResolution, thumbnailResolution), 0, 0);
            _thumb.Apply();
            RenderTexture.active = previousRT;

            // Compute average luminance over all pixels
            float luminanceSum = 0f;
            Color32[] pixels = _thumb.GetPixels32();
            foreach (var p in pixels)
                luminanceSum += 0.2126f * (p.r / 255f)
                              + 0.7152f * (p.g / 255f)
                              + 0.0722f * (p.b / 255f);

            float newBrightness = luminanceSum / pixels.Length;

            // Smoothed derivative: ḃ_t = (b_t - b_{t-1}) / Δt
            float rawRate = (newBrightness - _previousBrightness) / _sampleInterval;
            BrightnessRate = Mathf.Lerp(rawRate, BrightnessRate, derivativeSmoothing);
            Brightness = newBrightness;
            _previousBrightness = newBrightness;
        }


        private void OnGUI()
        {
#if UNITY_EDITOR
            GUI.Label(new Rect(10, 10, 300, 60),
                $"<b>Brightness</b> b_t = {Brightness:F4}\n" +
                $"<b>Rate</b> ḃ_t = {BrightnessRate:F4} /s",
                new GUIStyle { richText = true,
                               normal = { textColor = Color.white },
                               fontSize = 14 });
#endif
        }
    }
}
