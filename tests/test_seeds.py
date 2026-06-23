import pytest

from milp_dataset.seeds import derive_seed


def test_seed_is_deterministic_and_independent() -> None:
    first = derive_seed(2025, "ca", "train", 0)
    assert first == derive_seed(2025, "ca", "train", 0)
    identities = [
        ("ca", "train", 0),
        ("ca", "train", 1),
        ("ca", "valid", 0),
        ("sc", "train", 0),
    ]
    assert len({derive_seed(2025, *identity) for identity in identities}) == len(identities)


@pytest.mark.parametrize(
    "args", [(2025, "bad", "train", 0), (2025, "ca", "bad", 0), (2025, "ca", "train", -1)]
)
def test_seed_rejects_invalid_identity(args: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        derive_seed(*args)  # type: ignore[arg-type]
