// ═══════════════════════════════════════════════════════════════════════════════
//  SafeFallback.cs
//  When activated, steers the kart back to the designated safe start region
//  using the kart's existing IInput interface.  Both FORTRESS and Our Method
//  share this identical fallback logic — only the trigger condition differs.
// ═══════════════════════════════════════════════════════════════════════════════

using System.Collections;
using UnityEngine;

namespace OOD.Safety
{
    /// <summary>
    /// Shared fallback behaviour: navigates the kart to <see cref="safeTarget"/>
    /// (the brightly-lit start point) when an OOD controller raises an alert.
    /// <para>
    ///   Injects steering directly through Unity's input layer so it is
    ///   compatible with the existing ML-Agents <c>IInput</c> pipeline —
    ///   no changes to the kart controller are required.
    /// </para>
    /// </summary>
    [AddComponentMenu("OOD / Safety / Safe Fallback")]
    public sealed class SafeFallback : MonoBehaviour
    {

        [Header("Navigation Target")]
        [Tooltip("The safe start position the kart should return to.")]
        public Transform safeTarget;

        [Tooltip("Distance (m) at which the kart is considered 'arrived'.")]
        [Min(0.1f)] public float arrivalThreshold = 3f;

        [Header("Return Steering")]
        [Tooltip("Maximum steering angle applied during fallback (degrees).")]
        [Range(5f, 45f)] public float maxSteerAngle = 30f;

        [Tooltip("Forward speed applied when returning to safety (m/s).")]
        [Range(1f, 20f)] public float returnSpeed = 8f;

        [Tooltip("Proportional gain for heading correction.")]
        [Range(0.5f, 5f)] public float steerGain = 2.5f;

        [Header("Events")]
        public UnityEngine.Events.UnityEvent OnFallbackActivated;
        public UnityEngine.Events.UnityEvent OnSafeRegionReached;


        /// <summary>Whether the fallback is currently active.</summary>
        public bool IsActive { get; private set; }

        /// <summary>Whether the kart has successfully reached the safe target.</summary>
        public bool ReachedSafety { get; private set; }


        private Rigidbody _rb;


        private void Awake()
        {
            _rb = GetComponentInParent<Rigidbody>();

            if (safeTarget == null)
                Debug.LogWarning("[SafeFallback] No safe target assigned. " +
                                 "Assign the start-point transform in the Inspector.");
        }

        private void FixedUpdate()
        {
            if (!IsActive || ReachedSafety || safeTarget == null) return;

            NavigateToSafety();
            CheckArrival();
        }


        /// <summary>
        /// Activates the fallback. Call this from any OOD controller when risk
        /// exceeds the trigger threshold.
        /// </summary>
        public void Activate()
        {
            if (IsActive) return;

            IsActive = true;
            ReachedSafety = false;
            Debug.LogWarning("[SafeFallback] ⚠  OOD detected — returning to safe region.");
            OnFallbackActivated?.Invoke();
        }

        /// <summary>
        /// Deactivates the fallback and returns control to the ML agent.
        /// Called automatically on arrival; can also be called manually.
        /// </summary>
        public void Deactivate()
        {
            IsActive = false;
            Debug.Log("[SafeFallback] ✓ Fallback deactivated — agent control restored.");
        }


        private void NavigateToSafety()
        {
            // Direction to safe target in the kart's local XZ plane
            Vector3 toTarget = safeTarget.position - transform.position;
            toTarget.y = 0f;
            float distance = toTarget.magnitude;

            if (distance < 0.01f) return;

            Vector3 forward = transform.forward;
            forward.y = 0f;
            float signedAngle = Vector3.SignedAngle(forward, toTarget.normalized, Vector3.up);

            // Proportional steering clamped to max angle
            float steerInput = Mathf.Clamp(signedAngle * steerGain / maxSteerAngle, -1f, 1f);

            // Drive forward while correcting heading
            if (_rb != null)
            {
                _rb.MovePosition(_rb.position +
                                 transform.forward * returnSpeed * Time.fixedDeltaTime);
                _rb.MoveRotation(_rb.rotation *
                                 Quaternion.Euler(0f, steerInput * maxSteerAngle * Time.fixedDeltaTime, 0f));
            }
            else
            {
                // Fallback for non-rigidbody karts
                transform.position += transform.forward * returnSpeed * Time.fixedDeltaTime;
                transform.Rotate(0f, steerInput * maxSteerAngle * Time.fixedDeltaTime, 0f);
            }
        }

        private void CheckArrival()
        {
            float dist = Vector3.Distance(transform.position, safeTarget.position);
            if (dist > arrivalThreshold) return;

            ReachedSafety = true;
            IsActive = false;
            Debug.Log($"[SafeFallback] ✓ Safe region reached (dist = {dist:F2} m).");
            OnSafeRegionReached?.Invoke();
        }
    }
}
