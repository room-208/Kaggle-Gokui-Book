# Kaggle-Gokui-Book

## Usage
1. `docker compose`でコンテナを起動してください。
```sh
cd environments
docker compose up -d
```

2. [Kaggle](https://www.kaggle.com/)からAPIキーの`kaggle.json`をダウンロードし、コンテナ内の`/root/.kaggle`以下に配置してください。

3. [githubのcodeをgithub actionsの機能を使ってkaggle datasetにアップロードする](https://zenn.dev/hattan0523/articles/c55dfd51bb81e5)を参考に[dataset-metadata.json](./upload_kaggle_dir/dataset-metadata.json)を作成し、[Kaggle Dataset](https://www.kaggle.com/datasets)を作成してください。

4. [リポジトリに暗号化されたシークレットを作成する](https://docs.github.com/ja/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository)にしたがって[Kaggle](https://www.kaggle.com/)のAPIキーを登録してください。

5. `main`ブランチに`push`してください。`github`へのアップロード内容が自動的に[Kaggle Dataset](https://www.kaggle.com/datasets)へ反映されます。

## Reference
- [githubのcodeをgithub actionsの機能を使ってkaggle datasetにアップロードする](https://zenn.dev/hattan0523/articles/c55dfd51bb81e5)
- [kaggle-book-gokui](https://github.com/smly/kaggle-book-gokui)
- [Ascender](https://github.com/cvpaperchallenge/Ascender)