-- Token-bucket rate limiter.
--
-- KEYS[1] = bucket key (e.g. "rl:{owner}:{bucket}")
-- ARGV[1] = burst (max tokens)
-- ARGV[2] = refill_per_sec (float; tokens added per second)
-- ARGV[3] = now_ms (int; client clock, single source of truth)
-- ARGV[4] = cost (int; tokens to take)
--
-- Returns: { allowed (0/1), remaining (int), burst (int) }
--
-- State stored as a hash { tokens = float, last_ms = int }. Idle keys
-- expire after roughly twice the burst-refill window so dormant tenants
-- do not pile up.

local key             = KEYS[1]
local burst           = tonumber(ARGV[1])
local refill_per_sec  = tonumber(ARGV[2])
local now_ms          = tonumber(ARGV[3])
local cost            = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "last_ms")
local tokens  = tonumber(data[1])
local last_ms = tonumber(data[2])

if tokens == nil then
    tokens  = burst
    last_ms = now_ms
end

local elapsed_ms = math.max(0, now_ms - last_ms)
local refill = (elapsed_ms / 1000.0) * refill_per_sec
tokens = math.min(burst, tokens + refill)

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call("HMSET", key, "tokens", tokens, "last_ms", now_ms)

local ttl_ms = math.ceil((burst / math.max(refill_per_sec, 0.0001)) * 2000)
redis.call("PEXPIRE", key, ttl_ms)

return { allowed, math.floor(tokens), burst }
