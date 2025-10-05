"""Configuration defaults for SER estimation routines."""

DEFAULT_SER_PARAMETERS = {
    'build_years': 3,
    'avg_wage_base': 70_000,
    'tax_rates': {'pit': 0.13, 'cit': 0.17, 'prop': 0.02, 'land': 0.015},
    'va_coeff_build': {
        'BUSINESS': 0.50, 'RESIDENTIAL': 0.45, 'TRANSPORT': 0.55,
        'AGRICULTURE': 0.45, 'SPECIAL': 0.50, 'default': 0.50
    },
    'va_per_m2_ops': {
        'BUSINESS': 12000.0, 'RESIDENTIAL': 2000.0, 'TRANSPORT': 5000.0,
        'AGRICULTURE': 3000.0, 'SPECIAL': 7000.0, 'default': 0.0
    },
    'jobs_per_m2': {
        'BUSINESS': 1/18, 'RESIDENTIAL': 0.0, 'TRANSPORT': 1/90,
        'AGRICULTURE': 1/400, 'SPECIAL': 1/30, 'default': 0.0
    },
    'wage_by_use': {
        'BUSINESS': 85_000, 'RESIDENTIAL': 0, 'TRANSPORT': 60_000,
        'AGRICULTURE': 45_000, 'SPECIAL': 75_000, 'default': 0
    },
    'profit_share_ops': {
        'BUSINESS': 0.18, 'TRANSPORT': 0.10, 'AGRICULTURE': 0.08,
        'SPECIAL': 0.15, 'RESIDENTIAL': 0.00, 'default': 0.12
    },
    'capex_capitalizable_share': {
        'BUSINESS': 0.95, 'RESIDENTIAL': 0.95, 'TRANSPORT': 0.90,
        'AGRICULTURE': 0.90, 'SPECIAL': 0.95, 'default': 0.95
    },
    'amortization_rates': {
        'BUSINESS': 0.03, 'RESIDENTIAL': 0.03, 'TRANSPORT': 0.06,
        'AGRICULTURE': 0.05, 'SPECIAL': 0.04, 'default': 0.03
    },
    'build_wage_share': 0.25,
    'build_profit_margin': 0.05
}
