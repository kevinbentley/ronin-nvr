"""S3-compatible storage client for cold storage tier."""

import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)


class S3StorageClient:
    """Client for S3-compatible storage operations.

    Supports AWS S3, MinIO, Wasabi, and other S3-compatible services.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        bucket_name: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        prefix: str = "ronin-nvr/",
    ):
        """Initialize S3 client.

        Args:
            endpoint_url: S3 endpoint URL (None for AWS S3, or MinIO/Wasabi URL)
            bucket_name: S3 bucket name
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region (default us-east-1)
            prefix: Prefix for all S3 keys
        """
        settings = get_settings()

        self.endpoint_url = endpoint_url or settings.s3_endpoint_url
        self.bucket_name = bucket_name or settings.s3_bucket_name
        self.access_key = access_key or settings.s3_access_key
        self.secret_key = secret_key or settings.s3_secret_key
        self.region = region or settings.s3_region
        self.prefix = prefix or settings.s3_prefix

        self._client: Optional[boto3.client] = None

    @property
    def client(self) -> boto3.client:
        """Get or create boto3 S3 client."""
        if self._client is None:
            client_kwargs = {
                "service_name": "s3",
                "region_name": self.region,
            }

            if self.access_key and self.secret_key:
                client_kwargs["aws_access_key_id"] = self.access_key
                client_kwargs["aws_secret_access_key"] = self.secret_key

            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client(**client_kwargs)

        return self._client

    def _get_full_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        if key.startswith(self.prefix):
            return key
        return f"{self.prefix}{key}"

    def upload_file(
        self,
        local_path: Path,
        key: str,
        content_type: str = "video/mp4",
    ) -> str:
        """Upload a file to S3.

        Uses multipart upload for large files (handled automatically by boto3).

        Args:
            local_path: Local file path to upload
            key: S3 key (prefix will be added if not present)
            content_type: MIME type of the file

        Returns:
            Full S3 key of the uploaded file

        Raises:
            ClientError: If upload fails
        """
        full_key = self._get_full_key(key)

        try:
            extra_args = {"ContentType": content_type}

            # Use multipart upload config for large files
            config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=100 * 1024 * 1024,  # 100MB
                multipart_chunksize=100 * 1024 * 1024,  # 100MB chunks
                max_concurrency=4,
                use_threads=True,
            )

            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{full_key}")

            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                full_key,
                ExtraArgs=extra_args,
                Config=config,
            )

            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{full_key}")
            return full_key

        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise

    def generate_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned URL for direct access to an S3 object.

        Args:
            key: S3 key (prefix will be added if not present)
            expires_in: URL expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL for the object

        Raises:
            ClientError: If URL generation fails
        """
        full_key = self._get_full_key(key)

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": full_key,
                },
                ExpiresIn=expires_in,
            )
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {full_key}: {e}")
            raise

    def delete_file(self, key: str) -> bool:
        """Delete a file from S3.

        Args:
            key: S3 key (prefix will be added if not present)

        Returns:
            True if deleted successfully, False otherwise
        """
        full_key = self._get_full_key(key)

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            logger.info(f"Deleted s3://{self.bucket_name}/{full_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete {full_key} from S3: {e}")
            return False

    def check_exists(self, key: str) -> bool:
        """Check if a file exists in S3.

        Args:
            key: S3 key (prefix will be added if not present)

        Returns:
            True if the object exists, False otherwise
        """
        full_key = self._get_full_key(key)

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Failed to check existence of {full_key}: {e}")
            raise

    def get_object_size(self, key: str) -> Optional[int]:
        """Get the size of an S3 object in bytes.

        Args:
            key: S3 key (prefix will be added if not present)

        Returns:
            Size in bytes, or None if object doesn't exist
        """
        full_key = self._get_full_key(key)

        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=full_key,
            )
            return response["ContentLength"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            logger.error(f"Failed to get size of {full_key}: {e}")
            raise

    def is_configured(self) -> bool:
        """Check if S3 storage is properly configured.

        Returns:
            True if all required settings are present
        """
        return bool(
            self.bucket_name
            and self.access_key
            and self.secret_key
        )


# Global client instance (lazy initialized)
_s3_client: Optional[S3StorageClient] = None


def get_s3_client() -> S3StorageClient:
    """Get the global S3 client instance."""
    global _s3_client
    if _s3_client is None:
        _s3_client = S3StorageClient()
    return _s3_client
