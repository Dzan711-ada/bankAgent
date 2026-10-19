# 四大行公开银行卡资料整理包

整理日期：2026-10-12。第一批共 7 份：四家银行的借记卡资料，以及农业银行、中国银行、建设银行的信用卡资料。

本包提供原创的简短业务事实摘要、官方来源索引和项目兼容元数据；资料包未保存或转载章程全文，也不包含银行内部制度。摘要覆盖申领条件、材料、审核、代理/监护与启用等少量主题，不覆盖整份章程或全部办理流程。

## 文件

- `policies/*.md`：可上传的 UTF-8 正文；条目编号为整理编号，不对应原章程编号。
- `metadata/*.json`：与 `DocumentMetadata` 兼容；`document_type=public`、`content_kind=summary`、`reviewed=false`。
- `sources.json`：原文/公告/版权声明 URL、来源条款定位、版本、发布日期、生效日期与文件 SHA-256。
- `validation_report.json`：格式和条款解析检查；不表示业务或法律审核通过。

## 来源与日期

| 银行 / scope | 来源 | 来源制度生效日 |
|---|---|---|
| 工商银行 / `icbc` | [借记卡章程（2026年版）](https://media.icbc.com.cn/agreement/1279651951461629952.html)；[公告](https://www.icbc.com.cn/page/1222137992668459008.html) | 2026-06-01 |
| 农业银行 / `abc` | [金穗借记卡章程 ABC(2022)4001](https://www.abchina.com.cn/cn/PersonalServices/zcxy221101/zhkh221101/202301/P020230111446368301264.pdf) | 未确认；目录日期不能代替生效日 |
| 农业银行 / `abc` | [金穗信用卡章程（2023年版）](https://www.abchina.com/cn/CreditCard/AboutUs/Update/202304/W020230425333626683792.pdf)；[公告](https://www.abchina.com/cn/CreditCard/AboutUs/Update/202304/t20230425_2266670.htm) | 2023-06-17 |
| 中国银行 / `boc` | [长城借记卡章程（2021年版）](https://www.boc.cn/bcservice/bc3/bc31/202111/P020211109538431875935.pdf) | 未确认；保留相对“公布之日后”的原始表述定位 |
| 中国银行 / `boc` | [信用卡章程（2024年版）](https://www.boc.cn/bcservice/bi2/202402/P020240207374940847971.pdf)；[公告](https://www.boc.cn/bcservice/bi2/202402/t20240207_24560340.html) | 2024-03-31 |
| 建设银行 / `ccb` | [龙卡借记卡章程（2018年版）及公告](https://creditcard1.ccb.com/chn/2025-02/24/article_2025022414432244559.shtml) | 2018-08-15；URL 的 2025 不代表版本年份 |
| 建设银行 / `ccb` | [龙卡信用卡章程（2024年版）](https://creditcard.ccb.com/chn/creditcard/service/card_zhangcheng.shtml)；[公告](https://forex1.ccb.com/cn/creditcard/news/detail/20240719_1721378290.html) | 2024-09-02 |

元数据 `effective_date=2026-10-12` 是本次摘要在系统中的可用日期；原制度生效日另存为 `source_effective_date`，未确认的两份保持 `null`。它们不适合证明历史业务日期下的制度结论。审查人还应核对后续公告、适用卡种、地区、例外和柜面最新要求，不能仅凭搜索结果声称一定是现行版本。

## 使用与版权审核

四家银行的网站声明均对相关内容使用保留权利。本包没有取得针对你的用途的书面授权；原创短摘要和来源链接不等于已经获准商业发布、复制全文、镜像、电子抓取或其他受限使用。`sources.json` 统一保留 `rights_review_pending`，不标记为公有领域、CC0 或已获商业授权。

请结合你的实际用途核验所需许可或其他适用依据，尤其是商业问答部署以及将材料发送给模型服务的使用方式。摘要公开可访问并不保证任意用途合法。权利声明：[工行](https://www.icbc.com.cn/icbc/html/wzsm/wangzhanshengming_03.htm)、[农行](https://www.abchina.com/cn/PublicPlate/documents/201512/t20151228_817036.htm)、[中行](https://www.boc.cn/custserv/cs1/200812/t20081209_135606.html)、[建行](https://www.ccb.com/cn/v3/head_content/sitedeclare.html)。

原始整理模板保留 `reviewed=false`、审核人为空，供以后独立复核和复用。2026-10-12，用户明确指示直接导入后，本项目已将 7 份摘要加入本地知识库并发布，建立了真实嵌入索引。此次登记的是用户入库授权及自动来源事实核对，不表示银行审核或商业用途授权；版权状态仍为 `rights_review_pending`。

本项目的派生入库元数据、授权说明及发布记录位于 `reviewed/import_record.json`，四行非教学问答的真实接口检查位于 `reviewed/live_validation.json`。两类留档仅保存在当前项目，通用 ZIP 仍只包含原始待审核模板。当前项目已经完成本次导入，无需再次执行下面的导入命令。

## 检查与导入

在 `E:\bankAgent` 的 PyCharm 终端运行下面命令，默认检查不加载项目 `.env`、不调用模型接口，也不会写知识库：

```powershell
.\.venv\Scripts\python.exe scripts/import_public_package.py --check
```

完成业务核对和用途/授权核验后，可以运行以下命令。请将占位内容换成真实审核人和实际使用依据；`--publish` 会调用当前嵌入服务建立向量索引。

```powershell
.\.venv\Scripts\python.exe scripts/import_public_package.py --import-reviewed --reviewed-by "真实审核人或工号" --confirm-rights --rights-basis "实际授权编号或已核验的使用依据" --publish
```

脚本会保存已审核元数据及审核记录到本包 `reviewed/`，原整理包仍保留待审核状态。正文和元数据的校验值不匹配时会拒绝导入；修订资料需重新核对来源、更新版本和校验值。

也可通过网页逐份导入：左侧填管理员访问凭证 → 知识库管理 → 上传对应 `.md` → 按对应 JSON 填名称、版本及日期 → 资料类型选 `public`，内容类型选“整理摘要（非官方原文）” → 填来源制度生效日（未确认时留空） → 业务及用途审核完成后导入 → 发布。

问答时关闭“包括教学示例”，业务日期选择 2026-10-12 或之后，并在左侧“适用范围”填写 `icbc`、`abc`、`boc` 或 `ccb`。不要把银行简称填进“适用机构”：机构字段是用户身份权限，不是资料发行银行。

```powershell
.\.venv\Scripts\python.exe -m bank_agent.cli ask "工商银行借记卡需要哪些申请材料？" --scope icbc
```

第一次按银行分别测试，避免不指定范围时混合引用不同银行规则。答案和证据会明确标为“整理摘要（非官方原文）”，并提供官方来源链接。
