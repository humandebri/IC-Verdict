# 検証記録の管理

ソース、再生成スクリプト、小さな検証記録、集計結果はGitで管理します。
大きな測定JSONの盤面・手番・工程ごとの全履歴はローカルに保持し、
`.gitignore`の明示的なパス指定でGitから除外します。既存の追跡済み記録は変更しません。

対象は[summary-sources.json](summary-sources.json)に列挙しています。
[summaries/](summaries/)には実行条件、集計値、ゲーム単位の結果、元ファイルのSHA-256とサイズ、
省略した項目の位置・件数・ハッシュを保存します。要約にない詳細を検証するには生データが必要です。
ハッシュだけでは元データを復元できません。今回の整理では元ファイルを削除していません。

```sh
python3 tools/summarize_artifacts.py --write
python3 tools/summarize_artifacts.py --check
python3 tools/manifest.py --write
python3 tools/manifest.py --check
```

新しいcloneでは生データはありません。`--check`はその件数を明示し、存在する生データのみ照合します。
生データを作り直す方法は[Tetris入力調査](../docs/TETRIS_INPUT_STUDY.md)、
[query計測](../docs/QUERY_PROFILE.md)、各Tetris文書を参照してください。
入力調査の`fixtures.json`・`development.json`も生成物です。query調査の前に文書記載のnative調査を実行してください。
実測の時刻・応答時間などは変わるため、再実行で過去のハッシュが一致するとは限りません。

新たな大容量記録は自動的には除外しません。必要性を確認して対象一覧と`.gitignore`を更新し、
要約を生成します。生データを共有する場合は別の保管先を用意してください。
