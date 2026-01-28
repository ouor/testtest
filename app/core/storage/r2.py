from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import boto3
from botocore.client import Config

from app.core.errors.exceptions import AppError


@dataclass(frozen=True)
class R2Settings:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    endpoint_url: str

    @classmethod
    def from_env(cls) -> "R2Settings":
        account_id = (os.getenv("R2_ACCOUNT_ID") or "").strip()
        access_key_id = (os.getenv("R2_ACCESS_KEY_ID") or "").strip()
        secret_access_key = (os.getenv("R2_SECRET_ACCESS_KEY") or "").strip()
        bucket_name = (os.getenv("R2_BUCKET_NAME") or "").strip()

        endpoint_url = (os.getenv("R2_ENDPOINT_URL") or "").strip()
        if not endpoint_url and account_id:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

        missing: list[str] = []
        if not account_id and not endpoint_url:
            missing.append("R2_ACCOUNT_ID (or R2_ENDPOINT_URL)")
        if not access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not bucket_name:
            missing.append("R2_BUCKET_NAME")

        if missing:
            raise AppError(
                code="R2_CONFIG_MISSING",
                message="Missing Cloudflare R2 configuration",
                http_status=500,
                detail={"missing": missing},
            )

        return cls(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            endpoint_url=endpoint_url,
        )


def r2_enabled_from_env() -> bool:
    value = (os.getenv("R2_ENABLED", "0") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


class R2Storage:
    def __init__(self, *, settings: R2Settings) -> None:
        self.settings = settings
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    @classmethod
    def from_env(cls) -> "R2Storage":
        return cls(settings=R2Settings.from_env())

    @property
    def bucket(self) -> str:
        return self.settings.bucket_name

    def upload_bytes(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str | None = None,
        cache_control: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        if cache_control:
            extra["CacheControl"] = cache_control
        if metadata:
            extra["Metadata"] = metadata

        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def upload_file(self, *, path: str, key: str) -> None:
        self.client.upload_file(path, self.bucket, key)

    def download_bytes(self, *, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"]
        return body.read()

    def download_file(self, *, key: str, path: str) -> None:
        self.client.download_file(self.bucket, key, path)

    def delete(self, *, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, *, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_keys(self, *, prefix: str | None = None, limit: int = 1000) -> list[str]:
        if limit < 1:
            return []

        keys: list[str] = []
        continuation: str | None = None

        while True:
            params: dict[str, Any] = {"Bucket": self.bucket, "MaxKeys": min(limit - len(keys), 1000)}
            if prefix:
                params["Prefix"] = prefix
            if continuation:
                params["ContinuationToken"] = continuation

            resp = self.client.list_objects_v2(**params)
            contents = resp.get("Contents") or []
            for item in contents:
                k = item.get("Key")
                if k:
                    keys.append(str(k))
                    if len(keys) >= limit:
                        return keys

            if not resp.get("IsTruncated"):
                return keys

            continuation = resp.get("NextContinuationToken")
            if not continuation:
                return keys

    def presigned_get_url(self, *, key: str, expires_in: int = 86400) -> str:
        if expires_in < 1:
            raise ValueError("expires_in must be >= 1")

        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )
