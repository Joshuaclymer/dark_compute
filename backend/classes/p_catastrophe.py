import math

from backend.paramaters import PCatastropheParameters


class PCatastrophe:
    def _interpolate_probability(self, duration_years: float, p_1_month: float, p_1_year: float, p_10_years: float) -> float:
        """
        Interpolate probability using log-linear interpolation on duration.

        Uses the logit transform to ensure probabilities stay bounded (0, 1).
        Interpolates in logit space for smoother curves.

        Args:
            duration_years: The handoff duration in years
            p_1_month: Probability at 1 month (1/12 year)
            p_1_year: Probability at 1 year
            p_10_years: Probability at 10 years

        Returns:
            Interpolated probability bounded between 0 and 1
        """
        # Convert duration to log scale for interpolation
        # Time points in years
        t1 = 1/12  # 1 month
        t2 = 1.0   # 1 year
        t3 = 10.0  # 10 years

        # Clamp duration to avoid extrapolation issues
        duration_years = max(t1 / 10, min(duration_years, t3 * 10))

        log_t = math.log(duration_years)
        log_t1 = math.log(t1)
        log_t2 = math.log(t2)
        log_t3 = math.log(t3)

        # Convert probabilities to logit space for interpolation
        def logit(p):
            p = max(1e-10, min(1 - 1e-10, p))  # Clamp to avoid log(0)
            return math.log(p / (1 - p))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        logit_p1 = logit(p_1_month)
        logit_p2 = logit(p_1_year)
        logit_p3 = logit(p_10_years)

        # Piecewise linear interpolation in log-time, logit-probability space
        if log_t <= log_t2:
            # Interpolate between 1 month and 1 year
            alpha = (log_t - log_t1) / (log_t2 - log_t1)
            logit_p = logit_p1 + alpha * (logit_p2 - logit_p1)
        else:
            # Interpolate between 1 year and 10 years
            alpha = (log_t - log_t2) / (log_t3 - log_t2)
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
            params.p_ai_takeover_1_month,
            params.p_ai_takeover_1_year,
            params.p_ai_takeover_10_years
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
            params.p_human_power_grabs_1_month,
            params.p_human_power_grabs_1_year,
            params.p_human_power_grabs_10_years
        )