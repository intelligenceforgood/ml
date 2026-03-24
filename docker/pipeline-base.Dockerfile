# Base image for KFP pipeline components.
#
# Pre-installs kfp and all component-level dependencies so Vertex AI never
# runs pip inside a component container at startup.  This eliminates the two
# main sources of log noise:
#   1. Dependency conflicts: kfp==2.13.0 requires protobuf<5 but recent
#      google-cloud-aiplatform pulls protobuf>=6 when resolved independently.
#   2. "Running pip as root" warnings from KFP's injected install step.
#
# Binding constraint: kfp==2.13.0 requires protobuf>=4.21.1,<5.
# google-cloud-aiplatform<1.62 is the last series whose protobuf upper bound
# stays within <5; pin it here so the resolver stays within that range.
#
# Build:
#   scripts/build_image.sh pipeline-base dev

FROM python:3.11-slim

WORKDIR /app

# Silence the two most common pip noise sources so any remaining KFP-injected
# pip steps are quiet even when running as root inside the container.
ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Single solver pass: all packages resolved together avoids the protobuf
# version conflict that occurs when kfp and google-cloud-aiplatform are
# installed in separate pip invocations.
RUN pip install --no-cache-dir \
    "kfp==2.13.0" \
    "kfp-pipeline-spec==0.6.0" \
    "protobuf>=4.21.1,<5" \
    "google-cloud-aiplatform>=1.36,<1.62" \
    "google-cloud-storage>=2.0"
