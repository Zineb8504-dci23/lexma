import shutil, os

# Cloner le repo
os.system("git clone https://github.com/Zineb8504-dci23/lexma.git /content/lexma")

# Copier Qdrant depuis Drive vers le repo
shutil.copytree(
    "/content/drive/MyDrive/lexma_qdrant",
    "/content/lexma/data/qdrant"
)

# Push vers GitHub
os.chdir("/content/lexma")
os.system("git config user.email 'wardaghzineb@gmail.com'")
os.system("git config user.name 'Zineb8504-dci23'")
os.system("git add data/qdrant/")
os.system("git commit -m 'Add Qdrant database'")
os.system("git push https://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@github.com/Zineb8504-dci23/lexma.git") 
