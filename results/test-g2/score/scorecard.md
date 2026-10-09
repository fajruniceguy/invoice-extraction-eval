# Scorecard: run test-g2

- model: `deepseek/deepseek-v4-flash-vision-exp`
- prompt sha256: `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d`
- git: `unknown`
- gt hash: `e3a08b555dc4b98c01083688e4fd51de5d739ce65cc4d327f0f546a351ed2c76`
- dataset hash: `6dce9225448f84b731aab2a8523345c594b2f3a28389efcf3d7e2569e5dab643`
- scored: 25 document(s); unscored (no label): 0
- scored at: 2026-10-09T00:10:13+00:00

## Fields

| metric | value |
| --- | --- |
| labeled | 323 |
| correct | 290 |
| accuracy (abstentions are errors) | 0.898 |
| selective accuracy | 0.921 |
| coverage | 0.975 |
| abstention rate | 0.025 |
| no prediction rate | 0.000 |

| field | labeled | correct | accuracy | selective | coverage |
| --- | --- | --- | --- | --- | --- |
| vendor_name | 25 | 17 | 0.680 | 0.680 | 1.000 |
| vendor_address | 25 | 21 | 0.840 | 0.913 | 0.920 |
| vendor_tax_id | 25 | 25 | 1.000 | 1.000 | 1.000 |
| buyer_name | 25 | 25 | 1.000 | 1.000 | 1.000 |
| document_number | 25 | 22 | 0.880 | 0.880 | 1.000 |
| document_date | 24 | 21 | 0.875 | 0.875 | 1.000 |
| due_date | 25 | 25 | 1.000 | 1.000 | 1.000 |
| currency | 25 | 24 | 0.960 | 0.960 | 1.000 |
| payment_method | 25 | 24 | 0.960 | 0.960 | 1.000 |
| subtotal | 24 | 18 | 0.750 | 0.783 | 0.958 |
| tax | 25 | 24 | 0.960 | 1.000 | 0.960 |
| discount | 25 | 22 | 0.880 | 0.957 | 0.920 |
| total | 25 | 22 | 0.880 | 0.957 | 0.920 |

## Rows

- matched 169, missed 18, spurious 46, perfect 144
- precision 0.786, recall 0.904, F1 0.841, row accuracy 0.852, cell accuracy 0.944

## Errors

- numeric_mismatch: 7
- date_mismatch: 3
- string_mismatch: 9
- name_near_miss: 16
- hallucinated_value: 17
- missing_value: 8
- abstained_on_value: 7
- abstained_on_absent: 1
- row_missed: 18
- row_spurious: 46

## Slice: split

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| test | 323 | 0.898 | 0.921 | 0.975 |

## Slice: doc_type

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| faktur | 103 | 0.903 | 0.939 | 0.961 |
| hotel_reservation | 13 | 0.846 | 0.846 | 1.000 |
| bill | 13 | 0.846 | 0.846 | 1.000 |
| invoice | 39 | 0.949 | 0.974 | 0.974 |
| ecommerce | 39 | 0.872 | 0.895 | 0.974 |
| receipt | 116 | 0.897 | 0.912 | 0.983 |

## Slice: capture

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| born_digital | 116 | 0.879 | 0.919 | 0.957 |
| screen_capture | 39 | 0.923 | 0.923 | 1.000 |
| phone_photo | 142 | 0.887 | 0.906 | 0.979 |
| constructed | 26 | 1.000 | 1.000 | 1.000 |

## Cost and latency

Documents attempted (ok or failed); latency is over live calls only, nearest-rank.

| scope | docs | live | cached | paid | at list price | per doc (list) | tokens in/out | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 25 | 0 | 25 | $0.0000 | $0.1992 | $0.0080 | 115489/137108 | n/a | n/a |
| split=test | 25 | 0 | 25 | $0.0000 | $0.1992 | $0.0080 | 115489/137108 | n/a | n/a |
| doc_type=faktur | 8 | 0 | 8 | $0.0000 | $0.0362 | $0.0045 | 36957/20896 | n/a | n/a |
| doc_type=hotel_reservation | 1 | 0 | 1 | $0.0000 | $0.0041 | $0.0041 | 4559/2271 | n/a | n/a |
| doc_type=bill | 1 | 0 | 1 | $0.0000 | $0.0073 | $0.0073 | 3830/5147 | n/a | n/a |
| doc_type=invoice | 3 | 0 | 3 | $0.0000 | $0.0245 | $0.0082 | 17631/15980 | n/a | n/a |
| doc_type=ecommerce | 3 | 0 | 3 | $0.0000 | $0.0384 | $0.0128 | 13633/28560 | n/a | n/a |
| doc_type=receipt | 9 | 0 | 9 | $0.0000 | $0.0888 | $0.0099 | 38879/64254 | n/a | n/a |
| capture=born_digital | 9 | 0 | 9 | $0.0000 | $0.0529 | $0.0059 | 44985/32822 | n/a | n/a |
| capture=screen_capture | 3 | 0 | 3 | $0.0000 | $0.0149 | $0.0050 | 11429/9530 | n/a | n/a |
| capture=phone_photo | 11 | 0 | 11 | $0.0000 | $0.1110 | $0.0101 | 46951/80775 | n/a | n/a |
| capture=constructed | 2 | 0 | 2 | $0.0000 | $0.0204 | $0.0102 | 12124/13981 | n/a | n/a |

Unscored documents (no ground truth): none
