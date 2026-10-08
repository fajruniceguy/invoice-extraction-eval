# Scorecard: run test-g1

- model: `deepseek/deepseek-v4-flash-vision-exp`
- prompt sha256: `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d`
- git: `unknown`
- gt hash: `565e952cd59e8a863e28820f8710fb3bc262474df1db7fb6c53b87283cee0ee8`
- dataset hash: `e575a0f97887dc6c6da74d8f28a875d226369ce318bc9f767dcdcb04abf33fa4`
- scored: 25 document(s); unscored (no label): 0
- scored at: 2026-10-08T21:05:28+00:00

## Fields

| metric | value |
| --- | --- |
| labeled | 323 |
| correct | 273 |
| accuracy (abstentions are errors) | 0.845 |
| selective accuracy | 0.867 |
| coverage | 0.975 |
| abstention rate | 0.025 |
| no prediction rate | 0.000 |

| field | labeled | correct | accuracy | selective | coverage |
| --- | --- | --- | --- | --- | --- |
| vendor_name | 25 | 16 | 0.640 | 0.640 | 1.000 |
| vendor_address | 25 | 21 | 0.840 | 0.913 | 0.920 |
| vendor_tax_id | 25 | 25 | 1.000 | 1.000 | 1.000 |
| buyer_name | 25 | 22 | 0.880 | 0.880 | 1.000 |
| document_number | 25 | 19 | 0.760 | 0.760 | 1.000 |
| document_date | 24 | 20 | 0.833 | 0.833 | 1.000 |
| due_date | 25 | 25 | 1.000 | 1.000 | 1.000 |
| currency | 25 | 24 | 0.960 | 0.960 | 1.000 |
| payment_method | 25 | 23 | 0.920 | 0.920 | 1.000 |
| subtotal | 24 | 15 | 0.625 | 0.652 | 0.958 |
| tax | 25 | 21 | 0.840 | 0.875 | 0.960 |
| discount | 25 | 22 | 0.880 | 0.957 | 0.920 |
| total | 25 | 20 | 0.800 | 0.870 | 0.920 |

## Rows

- matched 163, missed 21, spurious 52, perfect 140
- precision 0.758, recall 0.886, F1 0.817, row accuracy 0.859, cell accuracy 0.945

## Errors

- numeric_mismatch: 10
- date_mismatch: 1
- string_mismatch: 13
- name_near_miss: 12
- hallucinated_value: 27
- missing_value: 12
- abstained_on_value: 6
- abstained_on_absent: 2
- row_missed: 21
- row_spurious: 52

## Slice: split

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| test | 323 | 0.845 | 0.867 | 0.975 |

## Slice: doc_type

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| faktur | 103 | 0.903 | 0.939 | 0.961 |
| hotel_reservation | 13 | 0.846 | 0.846 | 1.000 |
| bill | 52 | 0.519 | 0.540 | 0.962 |
| invoice | 39 | 0.949 | 0.974 | 0.974 |
| ecommerce | 13 | 0.923 | 0.923 | 1.000 |
| receipt | 103 | 0.903 | 0.912 | 0.990 |

## Slice: capture

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| born_digital | 116 | 0.879 | 0.919 | 0.957 |
| screen_capture | 39 | 0.923 | 0.923 | 1.000 |
| phone_photo | 142 | 0.768 | 0.784 | 0.979 |
| constructed | 26 | 1.000 | 1.000 | 1.000 |

## Cost and latency

Documents attempted (ok or failed); latency is over live calls only, nearest-rank.

| scope | docs | live | cached | paid | at list price | per doc (list) | tokens in/out | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 25 | 0 | 25 | $0.0000 | $0.1992 | $0.0080 | 115489/137108 | n/a | n/a |
| split=test | 25 | 0 | 25 | $0.0000 | $0.1992 | $0.0080 | 115489/137108 | n/a | n/a |
| doc_type=faktur | 8 | 0 | 8 | $0.0000 | $0.0362 | $0.0045 | 36957/20896 | n/a | n/a |
| doc_type=hotel_reservation | 1 | 0 | 1 | $0.0000 | $0.0041 | $0.0041 | 4559/2271 | n/a | n/a |
| doc_type=bill | 4 | 0 | 4 | $0.0000 | $0.0328 | $0.0082 | 15938/23385 | n/a | n/a |
| doc_type=invoice | 3 | 0 | 3 | $0.0000 | $0.0245 | $0.0082 | 17631/15980 | n/a | n/a |
| doc_type=ecommerce | 1 | 0 | 1 | $0.0000 | $0.0161 | $0.0161 | 5561/12039 | n/a | n/a |
| doc_type=receipt | 8 | 0 | 8 | $0.0000 | $0.0855 | $0.0107 | 34843/62537 | n/a | n/a |
| capture=born_digital | 9 | 0 | 9 | $0.0000 | $0.0529 | $0.0059 | 44985/32822 | n/a | n/a |
| capture=screen_capture | 3 | 0 | 3 | $0.0000 | $0.0149 | $0.0050 | 11429/9530 | n/a | n/a |
| capture=phone_photo | 11 | 0 | 11 | $0.0000 | $0.1110 | $0.0101 | 46951/80775 | n/a | n/a |
| capture=constructed | 2 | 0 | 2 | $0.0000 | $0.0204 | $0.0102 | 12124/13981 | n/a | n/a |

Unscored documents (no ground truth): none
