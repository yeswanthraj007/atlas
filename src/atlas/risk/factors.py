"""Time-series factor model: asset returns explained by factor returns.

    r_i,t = alpha_i + sum_k beta_i,k * f_k,t + e_i,t

Factor returns can be market / sector ETFs, Fama-French factors, or our own
cross-sectional style factors. The model gives:
  * factor exposures of the portfolio (beta-weighted dollars per factor)
  * a structured covariance  B F B' + D  that is far more stable than the sample matrix
  * a split of portfolio variance into factor risk and stock-specific risk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorModel:
    betas: pd.DataFrame  # assets x factors
    alphas: pd.Series  # per-period intercepts
    specific_var: pd.Series  # residual variance per asset
    factor_cov: pd.DataFrame  # factors x factors
    r_squared: pd.Series

    @property
    def assets(self) -> list[str]:
        return list(self.betas.index)

    def asset_covariance(self) -> pd.DataFrame:
        b = self.betas.to_numpy()
        cov = b @ self.factor_cov.to_numpy() @ b.T + np.diag(self.specific_var.to_numpy())
        return pd.DataFrame(cov, index=self.betas.index, columns=self.betas.index)

    def exposures(self, dollar_exposures: pd.Series) -> pd.Series:
        """Dollar exposure to each factor."""
        w = dollar_exposures.reindex(self.betas.index).fillna(0.0)
        return self.betas.T @ w

    def risk_decomposition(self, dollar_exposures: pd.Series) -> dict[str, float]:
        w = dollar_exposures.reindex(self.betas.index).fillna(0.0)
        fx = self.betas.T @ w
        factor_var = float(fx @ self.factor_cov @ fx)
        specific_var = float((w**2 * self.specific_var).sum())
        total = factor_var + specific_var
        return {
            "total_vol": float(np.sqrt(total)),
            "factor_vol": float(np.sqrt(factor_var)),
            "specific_vol": float(np.sqrt(specific_var)),
            "factor_share": factor_var / total if total else 0.0,
        }


def fit_factor_model(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> FactorModel:
    """OLS of every asset on the factors over their common dates."""
    joined = asset_returns.join(factor_returns, how="inner").dropna()
    if len(joined) <= factor_returns.shape[1] + 1:
        raise ValueError("not enough overlapping observations to fit the factor model")
    y = joined[asset_returns.columns].to_numpy()
    f = joined[factor_returns.columns].to_numpy()
    x = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    dof = len(joined) - x.shape[1]
    spec_var = (resid**2).sum(axis=0) / dof
    r2 = 1.0 - (resid**2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    assets, factors = asset_returns.columns, factor_returns.columns
    return FactorModel(
        betas=pd.DataFrame(coef[1:].T, index=assets, columns=factors),
        alphas=pd.Series(coef[0], index=assets),
        specific_var=pd.Series(spec_var, index=assets),
        factor_cov=pd.DataFrame(
            np.cov(f, rowvar=False, ddof=1).reshape(len(factors), -1),
            index=factors,
            columns=factors,
        ),
        r_squared=pd.Series(r2, index=assets),
    )
