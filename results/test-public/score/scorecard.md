# Scorecard: run test-public

- model: `deepseek/deepseek-v4-flash-vision-exp`
- prompt sha256: `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d`
- git: `dae4e0a401ea21623849f02e2b066ad0249ea9e8`
- gt hash: `32a7c3906f583966f01b53761fec7127dc6116ade00b8dcc74a4627707ddfb85`
- dataset hash: `bff330b9ba61afc2939526206b123edc41c4a2c237a32791fc7c068be99cb6f4`
- scored: 16 document(s); unscored (no label): 0
- scored at: 2026-10-14T05:48:44+00:00

## Fields

| metric | value |
| --- | --- |
| labeled | 207 |
| correct | 186 |
| accuracy (abstentions are errors) | 0.899 |
| selective accuracy | 0.916 |
| coverage | 0.981 |
| abstention rate | 0.019 |
| no prediction rate | 0.000 |

| field | labeled | correct | accuracy | selective | coverage |
| --- | --- | --- | --- | --- | --- |
| vendor_name | 16 | 10 | 0.625 | 0.625 | 1.000 |
| vendor_address | 16 | 14 | 0.875 | 0.933 | 0.938 |
| vendor_tax_id | 16 | 16 | 1.000 | 1.000 | 1.000 |
| buyer_name | 16 | 16 | 1.000 | 1.000 | 1.000 |
| document_number | 16 | 12 | 0.750 | 0.750 | 1.000 |
| document_date | 15 | 12 | 0.800 | 0.800 | 1.000 |
| due_date | 16 | 16 | 1.000 | 1.000 | 1.000 |
| currency | 16 | 15 | 0.938 | 0.938 | 1.000 |
| payment_method | 16 | 16 | 1.000 | 1.000 | 1.000 |
| subtotal | 16 | 16 | 1.000 | 1.000 | 1.000 |
| tax | 16 | 15 | 0.938 | 1.000 | 0.938 |
| discount | 16 | 14 | 0.875 | 0.933 | 0.938 |
| total | 16 | 14 | 0.875 | 0.933 | 0.938 |

## Rows

- matched 135, missed 10, spurious 48, perfect 116
- precision 0.738, recall 0.931, F1 0.823, row accuracy 0.859, cell accuracy 0.950

## Errors

- numeric_mismatch: 3
- date_mismatch: 3
- string_mismatch: 9
- name_near_miss: 9
- hallucinated_value: 14
- missing_value: 6
- abstained_on_value: 3
- abstained_on_absent: 1
- row_missed: 10
- row_spurious: 48

## Slice: split

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| test | 207 | 0.899 | 0.916 | 0.981 |

## Slice: doc_type

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| hotel_reservation | 13 | 1.000 | 1.000 | 1.000 |
| bill | 13 | 0.769 | 0.833 | 0.923 |
| ecommerce | 39 | 0.846 | 0.868 | 0.974 |
| receipt | 116 | 0.897 | 0.912 | 0.983 |
| invoice | 26 | 1.000 | 1.000 | 1.000 |

## Slice: capture

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| born_digital | 26 | 0.923 | 0.923 | 1.000 |
| screen_capture | 13 | 0.769 | 0.833 | 0.923 |
| phone_photo | 142 | 0.887 | 0.906 | 0.979 |
| constructed | 26 | 1.000 | 1.000 | 1.000 |

## Cost and latency

Documents attempted (ok or failed); latency is over live calls only, nearest-rank.

| scope | docs | live | cached | paid | at list price | per doc (list) | tokens in/out | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 16 | 2 | 14 | $0.0238 | $0.1606 | $0.0100 | 73025/115589 | 56.3 s | 66.9 s |
| split=test | 16 | 2 | 14 | $0.0238 | $0.1606 | $0.0100 | 73025/115589 | 56.3 s | 66.9 s |
| doc_type=hotel_reservation | 1 | 0 | 1 | $0.0000 | $0.0055 | $0.0055 | 4559/3403 | n/a | n/a |
| doc_type=bill | 1 | 1 | 0 | $0.0098 | $0.0098 | $0.0098 | 3830/7248 | 56.3 s | 56.3 s |
| doc_type=ecommerce | 3 | 1 | 2 | $0.0140 | $0.0362 | $0.0121 | 13633/26795 | 66.9 s | 66.9 s |
| doc_type=receipt | 9 | 0 | 9 | $0.0000 | $0.0888 | $0.0099 | 38879/64254 | n/a | n/a |
| doc_type=invoice | 2 | 0 | 2 | $0.0000 | $0.0203 | $0.0102 | 12124/13889 | n/a | n/a |
| capture=born_digital | 2 | 1 | 1 | $0.0140 | $0.0194 | $0.0097 | 10120/13677 | 66.9 s | 66.9 s |
| capture=screen_capture | 1 | 1 | 0 | $0.0098 | $0.0098 | $0.0098 | 3830/7248 | 56.3 s | 56.3 s |
| capture=phone_photo | 11 | 0 | 11 | $0.0000 | $0.1110 | $0.0101 | 46951/80775 | n/a | n/a |
| capture=constructed | 2 | 0 | 2 | $0.0000 | $0.0203 | $0.0102 | 12124/13889 | n/a | n/a |

Unscored documents (no ground truth): none
