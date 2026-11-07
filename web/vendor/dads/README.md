# デジタル庁デザインシステムのCSS

公式の[HTML版コードスニペット](https://github.com/digital-go-jp/design-system-example-components-html)から、以下のファイルを変更せず同梱している。
コミットは `caa50d731cf515cfa5eae0093121f11ac5299235` に固定した。
ライセンスは同ディレクトリの `LICENSE`（MIT）を参照。

| 同梱ファイル | 元ファイル |
| --- | --- |
| `global.css` | `src/global.css` |
| `button.css` | `src/components/button/button.css` |
| `textarea.css` | `src/components/textarea/textarea.css` |
| `select.css` | `src/components/select/select.css` |
| `form-control-label.css` | `src/components/form-control-label/form-control-label.css` |

使用したガイドライン：

- [ボタン](https://design.digital.go.jp/dads/components/button/)
- [テキストエリア](https://design.digital.go.jp/dads/components/textarea/)
- [タイポグラフィ](https://design.digital.go.jp/dads/foundations/typography/)

配色、文字サイズ、フォーカス表示、フォームラベルは公式CSSを使用する。
このページ固有のレイアウトは `web/app.css` に分離した。
外部通信を必要としないよう、Webフォントは取得せず、公式に許容されているOSのシステムフォントを使用する。
この翻訳ページはデジタル庁の提供するサービスではない。
