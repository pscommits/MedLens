"""
vault_agent.py
--------------
Encrypted report vault using Stellar keypairs + NaCl Box.

ENCRYPT (Doctor)
    Box(patient_x25519_pub, doctor_x25519_priv).encrypt(report_json)
    → saved to vault/{report_id}.enc
    → SHA-256 anchored on Stellar via ManageData

LOAD ALL REPORTS (Patient)
    Patient provides their own public key + secret key.
    Vault scans all .enc files matching that patient public key.
    Each is decrypted with Box(doctor_x25519_pub, patient_x25519_priv).
    Returns list of all their reports, newest first.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import base64
import hashlib
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import nacl.bindings
from nacl.public import PrivateKey, PublicKey, Box
from stellar_sdk import Keypair, Server, TransactionBuilder, Network
from stellar_sdk.exceptions import BaseHorizonError


_executor = ThreadPoolExecutor(max_workers=2)

VAULT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "vault"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

STELLAR_SECRET  = os.environ.get("STELLAR_SECRET_KEY", "").strip()
STELLAR_NETWORK = os.environ.get("STELLAR_NETWORK", "testnet").lower()

if STELLAR_NETWORK == "pubnet":
    HORIZON_URL        = "https://horizon.stellar.org"
    NETWORK_PASSPHRASE = Network.PUBLIC_NETWORK_PASSPHRASE
    EXPLORER_BASE      = "https://stellar.expert/explorer/public/tx"
else:
    HORIZON_URL        = "https://horizon-testnet.stellar.org"
    NETWORK_PASSPHRASE = Network.TESTNET_NETWORK_PASSPHRASE
    EXPLORER_BASE      = "https://stellar.expert/explorer/testnet/tx"


# ---------------------------------------------------------------------------
# Key conversion: Stellar Ed25519 → X25519 (Curve25519) for ECDH
# ---------------------------------------------------------------------------

def _stellar_secret_to_nacl_private(stellar_secret: str) -> PrivateKey:
    seed = Keypair.from_secret(stellar_secret).raw_secret_key()
    _, sk_64 = nacl.bindings.crypto_sign_seed_keypair(seed)
    x25519 = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(sk_64)
    return PrivateKey(x25519)


def _stellar_public_to_nacl_public(stellar_public: str) -> PublicKey:
    ed25519_pub = Keypair.from_public_key(stellar_public).raw_public_key()
    x25519 = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(ed25519_pub)
    return PublicKey(x25519)


def _validate_key(key: str, prefix: str) -> None:
    key = key.strip()
    if not key.startswith(prefix):
        label = "public" if prefix == "G" else "secret"
        raise ValueError(f"Stellar {label} key must start with '{prefix}', got '{key[:4]}...'")
    if len(key) != 56:
        raise ValueError(f"Stellar keys are 56 characters. Got {len(key)}.")


# ---------------------------------------------------------------------------
# Stellar ManageData anchor (best-effort)
# ---------------------------------------------------------------------------

def _anchor_to_stellar(content_hash: bytes, report_id: str) -> Optional[Tuple[str, str]]:
    if not STELLAR_SECRET:
        return None
    try:
        keypair = Keypair.from_secret(STELLAR_SECRET)
        server  = Server(horizon_url=HORIZON_URL)
        account = server.load_account(keypair.public_key)
        tx = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=NETWORK_PASSPHRASE,
                base_fee=100,
            )
            .append_manage_data_op(
                data_name=f"medlens_{report_id[:8]}",
                data_value=content_hash,
            )
            .set_timeout(30)
            .build()
        )
        tx.sign(keypair)
        response = server.submit_transaction(tx)
        tx_hash  = response["hash"]
        return tx_hash, f"{EXPLORER_BASE}/{tx_hash}"
    except BaseHorizonError as e:
        print(f"[vault_agent] Stellar anchor failed: {e.message}")
        return None
    except Exception as e:
        print(f"[vault_agent] Stellar anchor error: {e}")
        return None


# ---------------------------------------------------------------------------
# Vault file helpers
# ---------------------------------------------------------------------------

def _write_vault(report_id, ciphertext_b64, doctor_pub, patient_pub,
                 stellar_tx, stellar_explorer):
    record = {
        "report_id":             report_id,
        "encrypted_at":          time.time(),
        "doctor_stellar_pubkey": doctor_pub,
        "patient_stellar_pubkey": patient_pub,
        "stellar_tx_hash":       stellar_tx,
        "stellar_explorer":      stellar_explorer,
        "stellar_network":       STELLAR_NETWORK,
        "ciphertext_b64":        ciphertext_b64,
    }
    (VAULT_DIR / f"{report_id}.enc").write_text(json.dumps(record, indent=2))


def _read_vault(report_id: str) -> dict:
    path = VAULT_DIR / f"{report_id}.enc"
    if not path.is_file():
        raise FileNotFoundError(f"Report '{report_id}' not found in vault.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Synchronous core
# ---------------------------------------------------------------------------

def _encrypt_sync(analysis_dict: dict, patient_stellar_pubkey: str) -> dict:
    _validate_key(patient_stellar_pubkey, "G")
    if not STELLAR_SECRET:
        raise RuntimeError("STELLAR_SECRET_KEY not set in .env")

    doctor_pub = Keypair.from_secret(STELLAR_SECRET).public_key

    doctor_nacl_priv = _stellar_secret_to_nacl_private(STELLAR_SECRET)
    patient_nacl_pub = _stellar_public_to_nacl_public(patient_stellar_pubkey)

    box       = Box(doctor_nacl_priv, patient_nacl_pub)
    plaintext = json.dumps(analysis_dict, default=str).encode()
    ciphertext = box.encrypt(plaintext)

    report_id     = str(uuid.uuid4())
    content_hash  = hashlib.sha256(ciphertext).digest()
    stellar_result = _anchor_to_stellar(content_hash, report_id)
    stellar_tx     = stellar_result[0] if stellar_result else None
    stellar_exp    = stellar_result[1] if stellar_result else None

    _write_vault(
        report_id, base64.b64encode(ciphertext).decode(),
        doctor_pub, patient_stellar_pubkey, stellar_tx, stellar_exp,
    )

    return {
        "report_id":             report_id,
        "doctor_stellar_pubkey": doctor_pub,
        "patient_stellar_pubkey": patient_stellar_pubkey,
        "stellar_tx_hash":       stellar_tx,
        "stellar_explorer":      stellar_exp,
    }


def _get_patient_reports_sync(
    patient_stellar_secret: str,
) -> tuple[List[dict], str]:
    """
    Scan the vault for all reports encrypted for this patient.
    Derives the public key from the secret key automatically.
    Decrypt each one and return as a list (newest first) + the derived public key.
    """
    _validate_key(patient_stellar_secret, "S")

    # Derive public key from secret — patient only needs to provide their secret key
    derived_pub = Keypair.from_secret(patient_stellar_secret).public_key
    patient_stellar_pubkey = derived_pub

    patient_nacl_priv = _stellar_secret_to_nacl_private(patient_stellar_secret)

    reports = []
    vault_files = sorted(VAULT_DIR.glob("*.enc"),
                         key=lambda f: f.stat().st_mtime, reverse=True)

    for vault_file in vault_files:
        try:
            record = json.loads(vault_file.read_text())

            # Only include reports encrypted for this patient (matched by derived pubkey)
            if record.get("patient_stellar_pubkey") != patient_stellar_pubkey:
                continue

            doctor_pub  = record["doctor_stellar_pubkey"]
            ciphertext  = base64.b64decode(record["ciphertext_b64"])

            doctor_nacl_pub = _stellar_public_to_nacl_public(doctor_pub)
            box             = Box(patient_nacl_priv, doctor_nacl_pub)
            plaintext       = box.decrypt(ciphertext)
            analysis        = json.loads(plaintext.decode())

            # Re-attach vault metadata
            analysis["report_id"]             = record["report_id"]
            analysis["encrypted_at"]          = record.get("encrypted_at")
            analysis["doctor_stellar_pubkey"]  = doctor_pub
            analysis["patient_stellar_pubkey"] = patient_stellar_pubkey
            analysis["stellar_tx_hash"]        = record.get("stellar_tx_hash")
            analysis["stellar_explorer"]       = record.get("stellar_explorer")

            reports.append(analysis)

        except Exception as e:
            print(f"[vault_agent] Skipping {vault_file.name}: {e}")
            continue

    return reports, patient_stellar_pubkey


# ---------------------------------------------------------------------------
# Public async entry points
# ---------------------------------------------------------------------------

async def encrypt_and_store(analysis_dict: dict, patient_stellar_pubkey: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _encrypt_sync, analysis_dict, patient_stellar_pubkey
    )


async def get_patient_reports(
    patient_stellar_secret: str,
) -> tuple[List[dict], str]:
    """Return all reports encrypted for this patient (derived from secret key), newest first.
    Also returns the derived public key so callers don't need to pass it separately."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _get_patient_reports_sync,
        patient_stellar_secret,
    )