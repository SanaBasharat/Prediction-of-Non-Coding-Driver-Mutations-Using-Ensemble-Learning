import requests

"""
This Python script will download the gzipped hg19.fa and hg38.fa file from the UCSC Genome Browser 
and save it as hg19.fa.gz and hg38.fa.gz 

Please note: Downloading large files may take some time depending on your internet speed. 
Additionally, always ensure that you have the necessary permissions to download 
and use the data as per UCSC Genome Browser's terms and conditions.

"""
# Define the URL for the hg38.fa file
url_hg38 = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
url_hg19 = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"

# Define the local file name to save the downloaded files
# You can also change the file location as you desire
local_filename_hg38 = "hg38.fa.gz"
local_filename_hg19 = "hg19.fa.gz"

# Stream the download and save to file for hg38
with requests.get(url_hg38, stream=True) as response:
    with open(local_filename_hg38, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

print("Download complete for hg38 genome assembly fasta file!")


# Stream the download and save to file for hg19:

with requests.get(url_hg19, stream=True) as response:
    with open(local_filename_hg19, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

print("Download complete for hg19 genome assembly fasta file!")


