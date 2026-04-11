// ═══════════════════════════════════════════════════════════════════════════════
//  FORTRESSController.cs
//  Baseline: triggers the shared safe fallback whenever scene brightness falls
//  below a fixed threshold τ — no rate-of-change awareness.
//
//  Reference: FORTRESS (Fixed-threshold OOD Return to Safe State)
// ═══════════════════════════════════════════════════════════════════════════════

using UnityEngine;
using OOD.Perception;
using OOD.Safety;

namespace OOD.Controllers
{
    /// <summary>
    /// Implements the <b>FORTRESS</b> baseline controller.
    /// <para>
    ///   Trigger condition:
    ///   <code>  b_t &lt; τ  →  activate fallback  </code>
    ///   where <c>b_t</c> is current brightness and <c>τ</c> is the fixed
    ///   brightness threshold.  No derivative information is used.
    /// </para>
    /// </summary>
    [AddComponentMenu("OOD / Controllers / FORTRESS Controller")]
    [RequireComponent(typeof(BrightnessMonitor))]
    [RequireComponent(typeof(SafeFallback))]
    public sealed class FORTRESSController : MonoBehaviour
    {

        [Header("OOD Threshold")]
        [Tooltip("Brightness threshold τ.  When b_t drops below this value the " +
                 "fallback is triggered.  Must match Our Method's τ for a fair comparison.")]
        [Range(0f, 1f)] public float brightnessThreshold = 0.35f;

        [Header("Diagnostics")]
        [Tooltip("Log every evaluation tick to the console.")]
        public bool verboseLogging = false;


        /// <summary>Current OOD status according to FORTRESS.</summary>
        public bool IsOOD { get; private set; }


        private BrightnessMonitor _monitor;
        private SafeFallback _fallback;


        private void Awake()
        {
            _monitor = GetComponent<BrightnessMonitor>();
            _fallback = GetComponent<SafeFallback>();

            Debug.Log($"[FORTRESS] Initialised  τ = {brightnessThreshold:F3}");
        }

        private void Update()
        {
            if (_fallback.IsActive) return;

            float bt = _monitor.Brightness;
            IsOOD = bt < brightnessThreshold;

            if (verboseLogging)
                Debug.Log($"[FORTRESS]  b_t = {bt:F4}  τ = {brightnessThreshold:F3}  " +
                          $"OOD = {IsOOD}");

            if (IsOOD)
            {
                Debug.LogWarning($"[FORTRESS] ⚡ Threshold crossed — " +
                                 $"b_t ({bt:F4}) < τ ({brightnessThreshold:F3})");
                _fallback.Activate();
            }
        }


        private void OnGUI()
        {
#if UNITY_EDITOR
            float bt = _monitor != null ? _monitor.Brightness : 0f;

            // Colour the label red when OOD, green otherwise
            var style = new GUIStyle
            {
                richText = true,
                fontSize = 13,
                normal = { textColor = IsOOD ? Color.red : Color.green }
            };

            GUI.Label(new Rect(10, 80, 350, 80),
                $"<b>[FORTRESS]</b>\n" +
                $"  b_t = {bt:F4} τ = {brightnessThreshold:F3}\n" +
                $"  Status: {(IsOOD ? "⚠  OOD" : "✓  In-Distribution")}",
                style);
#endif
        }
    }
}
