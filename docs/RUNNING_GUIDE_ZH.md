# Activity-Aware Pipeline：运行说明

> 建议通过 `configs/default_config.json` 或 Windows 配置界面指定数据位置。`data/` 与 `outputs/` 仅提供目录结构，实际研究数据和运行结果默认不提交到 Git。

当前代码以已验证的路网拓扑为基础，并包含最终矢量几何平滑处理。

## 推荐运行方式

- 如果已有同一数据的稳定 S6 结果，可直接重新运行 S7。
- 如果希望全流程重新计算，可运行 `run_S2_to_S7_windows.cmd`。
- 最终平滑输出位于 S7 的 `08_final_clean_network/road_network_final_clean.*`。
- 平滑诊断：`08_final_clean_network/final_chain_smoothing.csv`。

## 路网平滑规则

1. 将两个真实路口/端点之间的完整 degree-2 路段作为一条连续道路链处理。
2. 中间 degree-2 节点可随曲线移动，真实交叉口和端点坐标固定。
3. 补断边参与同链平滑，不再在补断位置留下折角。
4. 普通道路默认强平滑，但必须落在 2.5 m 的已验证道路走廊内；失败时自动减弱。
5. U 型弯使用更窄的 0.8 m 保护走廊，避免把 hairpin 拉直。
6. 默认不再对平滑结果执行 Douglas-Peucker 强简化。
