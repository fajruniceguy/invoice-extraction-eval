# Scorecard: run test-final

- model: `deepseek/deepseek-v4-flash-vision-exp`
- prompt sha256: `7ec75dbb29063cbb5146cb5ee5ce6fcbedda0140ec39971c8f018f404126390d`
- git: `unknown`
- gt hash: `565e952cd59e8a863e28820f8710fb3bc262474df1db7fb6c53b87283cee0ee8`
- dataset hash: `e575a0f97887dc6c6da74d8f28a875d226369ce318bc9f767dcdcb04abf33fa4`
- scored: 25 document(s); unscored (no label): 0
- scored at: 2026-10-08T20:28:28+00:00

## Fields

| metric | value |
| --- | --- |
| labeled | 323 |
| correct | 256 |
| accuracy (abstentions are errors) | 0.793 |
| selective accuracy | 0.859 |
| coverage | 0.923 |
| abstention rate | 0.077 |
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
| subtotal | 24 | 10 | 0.417 | 0.556 | 0.750 |
| tax | 25 | 17 | 0.680 | 0.850 | 0.800 |
| discount | 25 | 21 | 0.840 | 0.955 | 0.880 |
| total | 25 | 13 | 0.520 | 0.812 | 0.640 |

## Rows

- matched 155, missed 29, spurious 60, perfect 137
- precision 0.721, recall 0.842, F1 0.777, row accuracy 0.884, cell accuracy 0.944

## Errors

- numeric_mismatch: 14
- date_mismatch: 1
- string_mismatch: 16
- name_near_miss: 9
- hallucinated_value: 24
- missing_value: 10
- abstained_on_value: 23
- abstained_on_absent: 2
- row_missed: 29
- row_spurious: 60

## Slice: split

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| test | 323 | 0.793 | 0.859 | 0.923 |

## Slice: doc_type

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| faktur | 103 | 0.903 | 0.939 | 0.961 |
| hotel_reservation | 13 | 0.769 | 0.833 | 0.923 |
| bill | 52 | 0.519 | 0.540 | 0.962 |
| invoice | 39 | 0.949 | 0.974 | 0.974 |
| ecommerce | 13 | 0.923 | 0.923 | 1.000 |
| receipt | 103 | 0.748 | 0.895 | 0.835 |

## Slice: capture

| value | labeled | accuracy | selective | coverage |
| --- | --- | --- | --- | --- |
| born_digital | 116 | 0.871 | 0.918 | 0.948 |
| screen_capture | 39 | 0.923 | 0.923 | 1.000 |
| phone_photo | 142 | 0.655 | 0.756 | 0.866 |
| constructed | 26 | 1.000 | 1.000 | 1.000 |

## Cost and latency

Documents attempted (ok or failed); latency is over live calls only, nearest-rank.

| scope | docs | live | cached | paid | at list price | per doc (list) | tokens in/out | latency p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 25 | 25 | 0 | $0.1992 | $0.1992 | $0.0080 | 115489/137108 | 12.5 s | 58.5 s |
| split=test | 25 | 25 | 0 | $0.1992 | $0.1992 | $0.0080 | 115489/137108 | 12.5 s | 58.5 s |
| doc_type=faktur | 8 | 8 | 0 | $0.0362 | $0.0362 | $0.0045 | 36957/20896 | 10.1 s | 18.1 s |
| doc_type=hotel_reservation | 1 | 1 | 0 | $0.0041 | $0.0041 | $0.0041 | 4559/2271 | 11.7 s | 11.7 s |
| doc_type=bill | 4 | 4 | 0 | $0.0328 | $0.0328 | $0.0082 | 15938/23385 | 12.5 s | 54.8 s |
| doc_type=invoice | 3 | 3 | 0 | $0.0245 | $0.0245 | $0.0082 | 17631/15980 | 20.2 s | 23.3 s |
| doc_type=ecommerce | 1 | 1 | 0 | $0.0161 | $0.0161 | $0.0161 | 5561/12039 | 48.3 s | 48.3 s |
| doc_type=receipt | 8 | 8 | 0 | $0.0855 | $0.0855 | $0.0107 | 34843/62537 | 10.4 s | 80.8 s |
| capture=born_digital | 9 | 9 | 0 | $0.0529 | $0.0529 | $0.0059 | 44985/32822 | 12.1 s | 48.3 s |
| capture=screen_capture | 3 | 3 | 0 | $0.0149 | $0.0149 | $0.0050 | 11429/9530 | 10.1 s | 20.9 s |
| capture=phone_photo | 11 | 11 | 0 | $0.1110 | $0.1110 | $0.0101 | 46951/80775 | 12.5 s | 80.8 s |
| capture=constructed | 2 | 2 | 0 | $0.0204 | $0.0204 | $0.0102 | 12124/13981 | 20.2 s | 23.3 s |

Unscored documents (no ground truth): none
