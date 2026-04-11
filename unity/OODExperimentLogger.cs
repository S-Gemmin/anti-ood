// ═══════════════════════════════════════════════════════════════════════════════
//  OODExperimentLogger.cs
//  Records per-frame telemetry for both controllers to a CSV so results can
//  be compared offline (e.g. in Python / matplotlib).
//
//  Output columns:
//    time, brightness, brightness_rate, fortress_ood, risk_value,
//    risk_triggered, fallback_active, reached_safety
// ═══════════════════════════════════════════════════════════════════════════════

using System;
using System.IO;
using System.Text;
using UnityEngine;
using OOD.Perception;
using OOD.Controllers;
using OOD.Safety;

namespace OOD.Logging
{
    /// <summary>
    /// Writes a CSV log of the experiment to
    /// <c>Application.persistentDataPath/ood_log_&lt;timestamp&gt;.csv</c>.
    /// Attach to the same GameObject as the other OOD components.
    /// </summary>
    [AddComponentMenu("OOD / Logging / Experiment Logger")]
    public sealed class OODExperimentLogger : MonoBehaviour
    {

        [Header("Logging")]
        [Tooltip("Log every N frames to keep file size manageable.")]
        [Min(1)] public int logEveryNFrames = 3;

        [Tooltip("Flush the CSV writer to disk every N seconds.")]
        [Min(1f)] public float flushInterval = 5f;

        [Header("Optional References")]
        [Tooltip("Leave null to auto-discover on the same GameObject.")]
        public BrightnessMonitor brightnessMonitor;
        public FORTRESSController fortressController;
        public ProactiveRiskController riskController;
        public SafeFallback safeFallback;


        private StreamWriter _writer;
        private int _frameCount;
        private float _nextFlush;
        private string _filePath;


        private void Awake()
        {
            AutoResolveReferences();

            _filePath = Path.Combine(Application.persistentDataPath,
                                     $"ood_log_{DateTime.Now:yyyyMMdd_HHmmss}.csv");

            _writer = new StreamWriter(_filePath, append: false, encoding: Encoding.UTF8);
            _writer.WriteLine("time,brightness,brightness_rate," +
                              "fortress_ood,risk_value,risk_triggered," +
                              "fallback_active,reached_safety");

            _nextFlush = Time.time + flushInterval;
            Debug.Log($"[OODLogger] Logging to: {_filePath}");
        }

        private void Update()
        {
            _frameCount++;
            if (_frameCount % logEveryNFrames != 0) return;

            WriteRow();

            if (Time.time >= _nextFlush)
            {
                _writer.Flush();
                _nextFlush = Time.time + flushInterval;
            }
        }

        private void OnDestroy()
        {
            if (_writer == null) return;
            _writer.Flush();
            _writer.Close();
            Debug.Log($"[OODLogger] Log finalised → {_filePath}");
        }


        private void WriteRow()
        {
            float time = Time.time;
            float bt = brightnessMonitor != null ? brightnessMonitor.Brightness : -1f;
            float dbt = brightnessMonitor != null ? brightnessMonitor.BrightnessRate : 0f;
            bool fortOOD = fortressController != null && fortressController.IsOOD;
            float riskVal = riskController != null ? riskController.CurrentRisk : 0f;
            bool riskTrig = riskController != null && riskController.IsTriggered;
            bool fbActive = safeFallback != null && safeFallback.IsActive;
            bool safe = safeFallback != null && safeFallback.ReachedSafety;

            _writer.WriteLine(
                $"{time:F4},{bt:F6},{dbt:F6}," +
                $"{BoolToInt(fortOOD)},{riskVal:F6},{BoolToInt(riskTrig)}," +
                $"{BoolToInt(fbActive)},{BoolToInt(safe)}");
        }

        private static int BoolToInt(bool v) => v ? 1 : 0;

        private void AutoResolveReferences()
        {
            if (brightnessMonitor == null) brightnessMonitor = GetComponent<BrightnessMonitor>();
            if (fortressController == null) fortressController = GetComponent<FORTRESSController>();
            if (riskController == null) riskController = GetComponent<ProactiveRiskController>();
            if (safeFallback == null) safeFallback = GetComponent<SafeFallback>();
        }
    }
}
