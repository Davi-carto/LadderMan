from ladder.utils.limb_heightmaps import heightmap_cache_is_current


def test_limb_heightmap_cache_is_shared_only_within_current_step():
    cache = {"common_step_counter": 7, "relative_height_m": object()}

    assert heightmap_cache_is_current(
        cache, common_step_counter=7, output="relative_height_m"
    )
    assert not heightmap_cache_is_current(
        cache, common_step_counter=8, output="relative_height_m"
    )
    assert not heightmap_cache_is_current(
        cache, common_step_counter=7, output="hit_valid"
    )
