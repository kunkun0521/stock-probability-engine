-- A股股票概率预测引擎 prob_* 表（文档 §9）
-- 全部 CREATE TABLE IF NOT EXISTS，Python 引擎 prepare 与 Java ProbSchemaInitializer 都会幂等执行。
-- 不要修改 stock_daily；字符集 utf8mb4；价格/概率用 decimal。
-- 注意：Java 端 resources/prob_schema.sql 与本文件保持完全一致。

CREATE TABLE IF NOT EXISTS prob_run (
  run_id BIGINT NOT NULL AUTO_INCREMENT,
  run_type VARCHAR(24) NOT NULL,
  asof_date DATE NULL,
  config_json JSON NULL,
  model_version VARCHAR(64) NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'RUNNING',
  n_universe INT NOT NULL DEFAULT 0,
  n_ok INT NOT NULL DEFAULT 0,
  n_fail INT NOT NULL DEFAULT 0,
  excluded_json JSON NULL,
  data_warn_count INT NOT NULL DEFAULT 0,
  started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at DATETIME NULL,
  message TEXT NULL,
  PRIMARY KEY (run_id),
  KEY idx_prob_run_type (run_type, started_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_prediction (
  code VARCHAR(10) NOT NULL,
  market CHAR(2) NOT NULL,
  asof_date DATE NOT NULL,
  horizon TINYINT NOT NULL,
  pred_type VARCHAR(8) NOT NULL,
  model_version VARCHAR(64) NOT NULL,
  run_id BIGINT NULL,
  p_up DECIMAL(8,6) NULL,
  p_down DECIMAL(8,6) NULL,
  exp_ret DECIMAL(10,6) NULL,
  median_ret DECIMAL(10,6) NULL,
  q05 DECIMAL(10,6) NULL,
  q25 DECIMAL(10,6) NULL,
  q50 DECIMAL(10,6) NULL,
  q75 DECIMAL(10,6) NULL,
  q95 DECIMAL(10,6) NULL,
  vol_1d_next DECIMAL(10,6) NULL,
  vol_h DECIMAL(10,6) NULL,
  prob_drop_5 DECIMAL(8,6) NULL,
  prob_drop_10 DECIMAL(8,6) NULL,
  prob_rise_5 DECIMAL(8,6) NULL,
  prob_rise_10 DECIMAL(8,6) NULL,
  var95 DECIMAL(10,6) NULL,
  var99 DECIMAL(10,6) NULL,
  cvar95 DECIMAL(10,6) NULL,
  risk_level TINYINT NULL,
  reliability_grade CHAR(1) NULL,
  components_json JSON NULL,
  selected_method VARCHAR(128) NULL,
  vol_model VARCHAR(16) NULL,
  flags VARCHAR(255) NULL,
  status VARCHAR(8) NOT NULL DEFAULT 'PENDING',
  settle_date DATE NULL,
  realized_ret DECIMAL(10,6) NULL,
  realized_up TINYINT NULL,
  hit_dir TINYINT NULL,
  below_q05 TINYINT NULL,
  above_q95 TINYINT NULL,
  breach_var95 TINYINT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (code, market, asof_date, horizon, pred_type, model_version),
  KEY idx_prob_pred_list (asof_date, horizon, pred_type, p_up),
  KEY idx_prob_pred_stock (code, market, horizon, asof_date),
  KEY idx_prob_pred_settle (status, horizon, asof_date),
  KEY idx_prob_pred_run (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_stock_model_state (
  code VARCHAR(10) NOT NULL,
  market CHAR(2) NOT NULL,
  asof_date DATE NOT NULL,
  vol_model VARCHAR(16) NULL,
  omega DECIMAL(14,8) NULL,
  alpha DECIMAL(14,8) NULL,
  beta DECIMAL(14,8) NULL,
  nu DECIMAL(10,4) NULL,
  loglik DECIMAL(16,4) NULL,
  converged TINYINT NULL,
  n_obs INT NULL,
  lb_pvalue DECIMAL(10,6) NULL,
  jb_pvalue DECIMAL(10,6) NULL,
  sigma_1d_next DECIMAL(14,8) NULL,
  vol_pctile_self DECIMAL(8,6) NULL,
  vol_pctile_mkt DECIMAL(8,6) NULL,
  markov_state VARCHAR(4) NULL,
  markov_matrix_json JSON NULL,
  markov_pvalue DECIMAL(10,6) NULL,
  mdd_250 DECIMAL(10,6) NULL,
  sharpe_250 DECIMAL(10,6) NULL,
  ann_vol_250 DECIMAL(10,6) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (code, market, asof_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_signal_stat (
  signal_key VARCHAR(128) NOT NULL,
  horizon TINYINT NOT NULL,
  train_end_date DATE NOT NULL,
  n INT NOT NULL DEFAULT 0,
  k INT NOT NULL DEFAULT 0,
  n_eff DECIMAL(14,4) NULL,
  p_hat DECIMAL(10,6) NULL,
  posterior_mean DECIMAL(10,6) NULL,
  ci_low DECIMAL(10,6) NULL,
  ci_high DECIMAL(10,6) NULL,
  p0 DECIMAL(10,6) NULL,
  prior_strength DECIMAL(10,4) NULL,
  half_life INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (signal_key, horizon, train_end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_model_registry (
  model_name VARCHAR(64) NOT NULL,
  horizon TINYINT NOT NULL,
  train_end_date DATE NOT NULL,
  artifact_path VARCHAR(255) NULL,
  params_json JSON NULL,
  metrics_json JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (model_name, horizon, train_end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_eval_metric (
  id BIGINT NOT NULL AUTO_INCREMENT,
  eval_date DATE NOT NULL,
  horizon TINYINT NOT NULL,
  scope VARCHAR(64) NOT NULL DEFAULT 'ALL',
  model_name VARCHAR(64) NOT NULL,
  metric_name VARCHAR(64) NOT NULL,
  value DECIMAL(18,8) NULL,
  ci_low DECIMAL(18,8) NULL,
  ci_high DECIMAL(18,8) NULL,
  n_samples INT NULL,
  base_value DECIMAL(18,8) NULL,
  delta DECIMAL(18,8) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_prob_eval (eval_date, horizon, scope, metric_name),
  KEY idx_prob_eval_model (model_name, metric_name, eval_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_calibration_bin (
  id BIGINT NOT NULL AUTO_INCREMENT,
  eval_date DATE NOT NULL,
  horizon TINYINT NOT NULL,
  model_name VARCHAR(64) NOT NULL,
  bin_no INT NOT NULL,
  bin_low DECIMAL(10,6) NULL,
  bin_high DECIMAL(10,6) NULL,
  n INT NULL,
  mean_pred DECIMAL(10,6) NULL,
  freq_up DECIMAL(10,6) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_prob_cal (eval_date, horizon, model_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_method_choice (
  id BIGINT NOT NULL AUTO_INCREMENT,
  eval_date DATE NOT NULL,
  horizon TINYINT NOT NULL,
  output_name VARCHAR(64) NOT NULL,
  selected_method VARCHAR(64) NOT NULL,
  score DECIMAL(18,8) NULL,
  candidates_json JSON NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_prob_choice (eval_date, horizon)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_backtest_summary (
  backtest_id BIGINT NOT NULL AUTO_INCREMENT,
  run_id BIGINT NULL,
  params_json JSON NULL,
  horizon TINYINT NOT NULL,
  start_date DATE NULL,
  end_date DATE NULL,
  annual_return DECIMAL(14,6) NULL,
  annual_vol DECIMAL(14,6) NULL,
  sharpe DECIMAL(14,6) NULL,
  max_drawdown DECIMAL(14,6) NULL,
  win_rate DECIMAL(10,6) NULL,
  win_pvalue DECIMAL(10,6) NULL,
  turnover DECIMAL(14,6) NULL,
  n_trades INT NULL,
  bench_annual_return DECIMAL(14,6) NULL,
  total_return DECIMAL(14,6) NULL,
  bench_total_return DECIMAL(14,6) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (backtest_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_backtest_nav (
  backtest_id BIGINT NOT NULL,
  trade_date DATE NOT NULL,
  nav DECIMAL(18,8) NOT NULL,
  bench_nav DECIMAL(18,8) NOT NULL,
  drawdown DECIMAL(12,6) NOT NULL,
  PRIMARY KEY (backtest_id, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prob_backtest_trade (
  id BIGINT NOT NULL AUTO_INCREMENT,
  backtest_id BIGINT NOT NULL,
  code VARCHAR(10) NOT NULL,
  market CHAR(2) NOT NULL,
  signal_date DATE NOT NULL,
  buy_date DATE NULL,
  sell_date DATE NULL,
  p_up DECIMAL(8,6) NULL,
  buy_price DECIMAL(12,4) NULL,
  sell_price DECIMAL(12,4) NULL,
  net_return DECIMAL(12,6) NULL,
  skipped VARCHAR(64) NULL,
  PRIMARY KEY (id),
  KEY idx_prob_bt_trade (backtest_id, signal_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
