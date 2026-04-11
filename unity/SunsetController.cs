// ═══════════════════════════════════════════════════════════════════════════════
//  SunsetController.cs
//  Drives the directional light intensity from 1.0 → 0.1 over a configurable
//  duration, simulating a sunset that pushes the scene out-of-distribution.
// ═══════════════════════════════════════════════════════════════════════════════

using UnityEngine;

namespace OOD.Environment
{
    /// <summary>
    /// Linearly attenuates a <see cref="Light"/> component's intensity to simulate
    /// a sunset, gradually shifting the scene beyond the safe daytime manifold.
    /// </summary>
    [AddComponentMenu("OOD / Environment / Sunset Controller")]
    [RequireComponent(typeof(Light))]
    public sealed class SunsetController : MonoBehaviour
    {

        [Header("Intensity Curve")]
        [Tooltip("Starting light intensity — represents full daylight.")]
        [Range(0f, 8f)] public float intensityStart = 1.0f;

        [Tooltip("Ending light intensity — 10 % of full daylight.")]
        [Range(0f, 8f)] public float intensityEnd = 0.1f;

        [Tooltip("Time in seconds to fade from start to end intensity.")]
        [Min(1f)] public float sunsetDuration = 50f;

        [Header("Playback")]
        [Tooltip("Begin fading immediately on Start.")]
        public bool autoPlay = true;


        /// <summary>Normalised [0 – 1] progress through the sunset.</summary>
        public float Progress => _elapsed / sunsetDuration;

        /// <summary>Whether the sunset animation has completed.</summary>
        public bool Finished => _elapsed >= sunsetDuration;


        private Light _sun;
        private float _elapsed;
        private bool _running;


        private void Awake()
        {
            _sun = GetComponent<Light>();
            _sun.intensity = intensityStart;
        }

        private void Start()
        {
            if (autoPlay) BeginSunset();
        }

        private void Update()
        {
            if (!_running) return;

            _elapsed = Mathf.Min(_elapsed + Time.deltaTime, sunsetDuration);
            _sun.intensity = Mathf.Lerp(intensityStart, intensityEnd,
                                        _elapsed / sunsetDuration);

            if (Finished)
            {
                _running = false;
                Debug.Log("[SunsetController] Sunset complete.");
            }
        }


        /// <summary>Starts (or restarts) the sunset animation.</summary>
        public void BeginSunset()
        {
            _elapsed = 0f;
            _running = true;
            _sun.intensity = intensityStart;
            Debug.Log("[SunsetController] Sunset begun.");
        }

        /// <summary>Pauses the animation at its current point.</summary>
        public void Pause() => _running = false;

        /// <summary>Resumes a paused animation.</summary>
        public void Resume() => _running = true;

        /// <summary>Immediately jumps to the fully dark end-state.</summary>
        public void SkipToEnd()
        {
            _elapsed = sunsetDuration;
            _sun.intensity = intensityEnd;
            _running = false;
        }
    }
}
