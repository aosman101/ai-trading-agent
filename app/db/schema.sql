create table if not exists predictions (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    symbol text not null,
    direction text not null,
    confidence double precision,
    weighted_score double precision,
    selected_strategy text,
    most_influential_model text,
    explanation text,
    payload jsonb
);

create table if not exists trades (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    symbol text not null,
    direction text not null,
    quantity integer,
    entry_price double precision,
    stop_loss double precision,
    take_profit double precision,
    risk_amount double precision,
    notional double precision,
    broker_order_id text,
    status text,
    strategy text,
    payload jsonb
);

create table if not exists equity_curve (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    equity double precision,
    day_pnl double precision,
    payload jsonb
);

create table if not exists learning_events (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    event_type text,
    message text,
    payload jsonb
);

create table if not exists model_weights (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    payload jsonb
);

create table if not exists runtime_state (
    state_key text primary key,
    updated_at timestamptz default now(),
    payload jsonb
);

create table if not exists external_signals (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    symbol text not null,
    direction text not null,
    score double precision default 0.0,
    confidence double precision default 0.5,
    source text default 'website',
    reasoning text,
    consumed_at timestamptz,
    payload jsonb
);

create table if not exists journal (
    id bigint generated always as identity primary key,
    created_at timestamptz default now(),
    symbol text,
    event_type text not null,
    headline text not null,
    body text not null,
    payload jsonb
);
