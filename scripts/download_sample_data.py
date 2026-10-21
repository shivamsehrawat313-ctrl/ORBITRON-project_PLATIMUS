"""Print official SpaceNet-2 Paris sample retrieval instructions; no download occurs."""
print("Review the official SpaceNet-2 page and CC BY-SA 4.0 terms first:")
print("https://spacenet.ai/spacenet-buildings-dataset-v2/")
print("The Paris training archive is about 5.3 GB and may require an AWS account.")
print("PowerShell (after installing/configuring AWS CLI):")
print("New-Item -ItemType Directory -Force data/sample | Out-Null")
print("aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz data/sample/")
print("This script does not scrape or download imagery.")
