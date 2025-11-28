import math

from backend.paramaters import PCatastropheParameters


class PCatastrophe:
    # Time anchor points in years - change these to adjust the interpolation points
    ANCHOR_T1 = 1/12  # 1 month (first anchor)
    ANCHOR_T2 = 1.0   # 1 year (second anchor)
    ANCHOR_T3 = 10.0  # 10 years (third anchor)

    @staticmethod
    def compute_safety_speedup(capability_speedup: float, exponent: float) -> float:
        """
        Compute safety research speedup from capability research speedup.

        Safety research speedup = capability_speedup ^ exponent

        For example, if capability speedup is 4x and exponent is 0.5,
        safety speedup is sqrt(4) = 2x.

        Args:
            capability_speedup: The AI capability research speedup factor
            exponent: The exponent relating safety to capability speedup

        Returns:
            Safety research speedup factor
        """
        if capability_speedup <= 0:
            return 1.0
        return capability_speedup ** exponent

    def _interpolate_probability(self, duration_years: float, p_t1: float, p_t2: float, p_t3: float) -> float:
        """
        Interpolate probability using log-linear interpolation on duration.

        Uses the logit transform to ensure probabilities stay bounded (0, 1).
        Interpolates in logit space for smoother curves, with log-time for x-axis.

        Args:
            duration_years: The handoff duration in years
            p_t1: Probability at ANCHOR_T1 (default: 1 month)
            p_t2: Probability at ANCHOR_T2 (default: 1 year)
            p_t3: Probability at ANCHOR_T3 (default: 10 years)

        Returns:
            Interpolated probability bounded between 0 and 1
        """
        # Time anchor points in years
        t1 = self.ANCHOR_T1
        t2 = self.ANCHOR_T2
        t3 = self.ANCHOR_T3

        # Clamp duration to valid range (minimum is t1 for log interpolation)
        duration_years = max(t1, min(duration_years, t3 * 10))

        # Convert probabilities to logit space for interpolation
        def logit(p):
            p = max(1e-10, min(1 - 1e-10, p))  # Clamp to avoid log(0)
            return math.log(p / (1 - p))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        logit_p1 = logit(p_t1)
        logit_p2 = logit(p_t2)
        logit_p3 = logit(p_t3)

        # Use log-time for interpolation (log-linear in time)
        log_t1 = math.log(t1)
        log_t2 = math.log(t2)
        log_t3 = math.log(t3)
        log_duration = math.log(duration_years)

        # Piecewise linear interpolation in log-time, logit-probability space
        if log_duration <= log_t2:
            # Interpolate between 1 month and 1 year
            alpha = (log_duration - log_t1) / (log_t2 - log_t1)
            logit_p = logit_p1 + alpha * (logit_p2 - logit_p1)
        else:
            # Interpolate between 1 year and 10 years
            alpha = (log_duration - log_t2) / (log_t3 - log_t2)
            logit_p = logit_p2 + alpha * (logit_p3 - logit_p2)

        return sigmoid(logit_p)

    def p_AI_takeover(self, research_speed_adjusted_handoff_duration: float, params: PCatastropheParameters) -> float:
        """
        Estimate the probability of AI takeover given the adjusted takeoff duration.

        Args:
            research_speed_adjusted_handoff_duration (float): The handoff duration is the duration of time between SC and SAR. The research speed adjusted handoff duration is the handoff duration adjusted for the rate of AI R&D speedup. So for example, if the rate of AI R&D speed up is 2x during a handoff duration of 1 year, then the research speed adjusted handoff duration would be 2 years.
        Returns:
            float: Estimated probability of AI takeover.
        """
        return self._interpolate_probability(
            research_speed_adjusted_handoff_duration,
            params.p_ai_takeover_t1,
            params.p_ai_takeover_t2,
            params.p_ai_takeover_t3
        )

    def p_human_power_grabs(self, handoff_duration: float, params: PCatastropheParameters) -> float:
        """
        Estimate the probability of human power grabs given the handoff duration.

        Args:
            handoff_duration (float): The handoff duration is the duration of time between SC and SAR.
        Returns:
            float: Estimated probability of human power grabs
        """
        return self._interpolate_probability(
            handoff_duration,
            params.p_human_power_grabs_t1,
            params.p_human_power_grabs_t2,
            params.p_human_power_grabs_t3
        )

    def p_domestic_takeover(
        self,
        handoff_duration: float,
        research_speed_adjusted_handoff_duration: float,
        params: PCatastropheParameters
    ) -> float:
        """
        Compute the combined probability of catastrophe from AI takeover and human power grabs.

        Assumes independence between the two risks:
        P(catastrophe) = 1 - (1 - P(AI takeover)) * (1 - P(human power grabs))

        Args:
            handoff_duration: Raw handoff duration (SC to SAR) in years
            research_speed_adjusted_handoff_duration: Handoff duration adjusted for AI R&D speedup
            params: PCatastropheParameters with probability anchor points

        Returns:
            Combined probability of catastrophe
        """
        p_takeover = self.p_AI_takeover(research_speed_adjusted_handoff_duration, params)
        p_power_grabs = self.p_human_power_grabs(handoff_duration, params)
        return 1 - (1 - p_takeover) * (1 - p_power_grabs)