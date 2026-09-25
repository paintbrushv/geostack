-- Add the Texas PTAD state-use category code to the parcels table.
-- normalize_parcels() now emits `state_use_code` for every county that exposes a
-- PTAD code (Denton imprv_state_cd, Collin propcategorycode, Grayson imprv_state_cd),
-- so load_parcels_to_postgis appends that column — it must exist in the relation.
-- The legacy `property_category` column is retained for back-compat (now unused).

ALTER TABLE parcels ADD COLUMN IF NOT EXISTS state_use_code TEXT;
