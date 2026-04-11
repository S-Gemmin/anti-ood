// ═══════════════════════════════════════════════════════════════════════════════
//  ProactiveRiskController.cs
//  "Our Method" — rate-aware OOD controller.
//
//  Risk is computed as:
//
//      risk = (τ − b_t) + β · max(0, −ḃ_t)
//
//  where b_t is current brightness, ḃ_t is its rate of change, τ is the
//  brightness threshold, and β weights the velocity penalty.
//  Fallback triggers when risk > ε (riskTrigger).
// ═══════════════════════════════════════════════════════════════════════════════

using UnityEngine;
using System.Collections.Generic;
using OOD.Perception;
using OOD.Safety;

namespace OOD.Controllers
{
    /// <summary>
    /// Implements the proactive rate-aware OOD controller ("Our Method").
    /// <para>
    ///   Unlike <see cref="FORTRESSController"/>, this controller also penalises
    ///   negative brightness trends (<c>ḃ_t &lt; 0</c>), enabling earlier
    ///   intervention before the hard threshold is crossed.
    /// </para>
    /// <para>
    ///   Risk formula:
    ///   <code>  risk = (τ − b_t)  +  β · max(0, −ḃ_t)  </code>
    ///   Fallback activates when <c>risk &gt; ε</c>.
    /// </para>
    /// </summary>
    [AddComponentMenu("OOD / Controllers / Proactive Risk Controller")]
    [RequireComponent(typeof(BrightnessMonitor))]
    [RequireComponent(typeof(SafeFallback))]
    public sealed class ProactiveRiskController : MonoBehaviour
    {

        [Header("Risk Parameters")]
        [Tooltip("Brightness threshold τ — the same value used by FORTRESS for a fair comparison.")]
        [Range(0f, 1f)] public float brightnessThreshold = 0.35f;

        [Tooltip("Velocity penalty weight β.  Higher values make the controller " +
                 "react earlier to rapid darkening.")]
        [Range(0f, 10f)] public float beta = 3.0f;

        [Tooltip("Risk trigger threshold ε.  Fallback activates when risk > ε.")]
        [Range(0f, 1f)] public float riskTrigger = 0.1f;

        [Header("Diagnostics")]
        [Tooltip("Log every evaluation tick to the console.")]
        public bool verboseLogging = false;

        [Tooltip("Number of risk history samples shown on the debug graph.")]
        [Range(20, 200)] public int historyLength = 100;


        /// <summary>Most recently computed risk value.</summary>
        public float CurrentRisk { get; private set; }

        /// <summary>Whether risk currently exceeds the trigger threshold.</summary>
        public bool IsTriggered { get; private set; }


        private BrightnessMonitor _monitor;
        private SafeFallback _fallback;
        private readonly Queue<float> _riskHistory = new Queue<float>();


        private void Awake()
        {
            _monitor = GetComponent<BrightnessMonitor>();
            _fallback = GetComponent<SafeFallback>();

            Debug.Log($"[ProactiveRisk] Initialised — " +
                      $"τ = {brightnessThreshold:F3}  β = {beta:F2}  ε = {riskTrigger:F3}");
        }

        private void Update()
        {
            if (_fallback.IsActive) return;

            float bt = _monitor.Brightness;
            float dbt = _monitor.BrightnessRate;

            //   risk = (τ − b_t)  +  β · max(0, −ḃ_t)
            float proximityTerm = brightnessThreshold - bt;
            float velocityPenalty = beta * Mathf.Max(0f, -dbt);
            CurrentRisk = proximityTerm + velocityPenalty;
            IsTriggered = CurrentRisk > riskTrigger;

            // Track history for editor graph
            _riskHistory.Enqueue(CurrentRisk);
            while (_riskHistory.Count > historyLength)
                _riskHistory.Dequeue();

            if (verboseLogging)
                Debug.Log($"[ProactiveRisk]  b_t={bt:F4}  ḃ_t={dbt:F4}  " +
                          $"proximity={proximityTerm:F4}  velocity={velocityPenalty:F4}  " +
                          $"risk={CurrentRisk:F4}  triggered={IsTriggered}");

            if (IsTriggered)
            {
                Debug.LogWarning($"[ProactiveRisk] ⚡ Risk threshold exceeded — " +
                                 $"risk ({CurrentRisk:F4}) > ε ({riskTrigger:F3})\n" +
                                 $"  b_t={bt:F4}  ḃ_t={dbt:F4}  β·|ḃ_t|={velocityPenalty:F4}");
                _fallback.Activate();
            }
        }


        private void OnGUI()
        {
#if UNITY_EDITOR
            float bt = _monitor != null ? _monitor.Brightness : 0f;
            float dbt = _monitor != null ? _monitor.BrightnessRate : 0f;

            var style = new GUIStyle
            {
                richText = true,
                fontSize = 13,
                normal   = { textColor = IsTriggered ? Color.red : Color.cyan }
            };

            GUI.Label(new Rect(10, 170, 400, 120),
                $"<b>[Proactive Risk Controller]</b>\n" +
                $"  b_t = {bt:F4}\n" +
                $"  ḃ_t = {dbt:F4} /s\n" +
                $"  risk = ({brightnessThreshold:F3} − {bt:F4}) " +
                    $"+ {beta:F1}·max(0,{-dbt:F4})\n" +
                $"        = <b>{CurrentRisk:F4}</b> (ε = {riskTrigger:F3})\n" +
                $"  Status: {(IsTriggered ? "⚠  TRIGGERED" : "✓  Safe")}",
                style);

            DrawRiskGraph();
#endif
        }

#if UNITY_EDITOR
        /// <summary>Renders a minimal sparkline of recent risk values in the corner.</summary>
        private void DrawRiskGraph()
        {
            const float gx = 10f, gy = 300f, gw = 200f, gh = 60f;

            // Background box
            GUI.color = new Color(0f, 0f, 0f, 0.5f);
            GUI.DrawTexture(new Rect(gx - 2, gy - 2, gw + 4, gh + 4),
                            Texture2D.whiteTexture);
            GUI.color = Color.white;

            float[] history = new float[_riskHistory.Count];
            _riskHistory.CopyTo(history, 0);
            if (history.Length < 2) return;

            float maxRisk = Mathf.Max(riskTrigger * 2f, 0.01f);

            for (int i = 1; i < history.Length; i++)
            {
                float x0 = gx + (i - 1f) / historyLength * gw;
                float x1 = gx + i / historyLength * gw;
                float y0 = gy + gh - (history[i - 1] / maxRisk) * gh;
                float y1 = gy + gh - (history[i] / maxRisk) * gh;
                y0 = Mathf.Clamp(y0, gy, gy + gh);
                y1 = Mathf.Clamp(y1, gy, gy + gh);

                // Draw a thin line segment via a rotated texture
                Vector2 dir = new Vector2(x1 - x0, y1 - y0);
                float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
                float len = dir.magnitude;

                var mat = GUI.matrix;
                GUIUtility.RotateAroundPivot(angle, new Vector2(x0, y0));
                GUI.color = history[i] > riskTrigger ? Color.red : Color.cyan;
                GUI.DrawTexture(new Rect(x0, y0 - 1f, len, 2f), Texture2D.whiteTexture);
                GUI.matrix = mat;
                GUI.color = Color.white;
            }

            // Trigger-threshold line
            float ty = gy + gh - (riskTrigger / maxRisk) * gh;
            GUI.color = new Color(1f, 0.5f, 0f, 0.8f);
            GUI.DrawTexture(new Rect(gx, ty, gw, 1f), Texture2D.whiteTexture);
            GUI.color = Color.white;

            GUI.Label(new Rect(gx + 2, gy + gh - 15, 80, 14),
                      "Risk history",
                      new GUIStyle { fontSize = 10, normal = { textColor = Color.gray } });
        }
#endif
    }
}
