-- Run once in Supabase SQL Editor (https://supabase.com/dashboard/project/<id>/sql)
-- Creates the cache table used by src/cache.py.
--
-- The cache is OPTIONAL: src/cache.py gracefully no-ops if the table is missing
-- or USE_SUPABASE_CACHE=false in .env.

create table if not exists query_cache (
    question_hash text primary key,        -- sha256(lower(trim(question)))
    question      text not null,
    response      jsonb not null,
    created_at    timestamptz not null default now()
);

create index if not exists query_cache_created_idx on query_cache(created_at desc);

-- RLS: lock down to service-role writes only (anon role is read-blocked anyway,
-- but explicit is better than implicit). Server uses service_role key.
alter table query_cache enable row level security;

drop policy if exists "service_role full access" on query_cache;
create policy "service_role full access" on query_cache
    for all using (auth.role() = 'service_role') with check (auth.role() = 'service_role');
