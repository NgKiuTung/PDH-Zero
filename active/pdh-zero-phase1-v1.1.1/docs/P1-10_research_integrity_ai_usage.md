# P1-10 研究诚信与生成式 AI 使用记录

本项目面向学术投稿，所有模型辅助必须可追溯。建议在项目根目录维护：

```text
docs/ai_usage_log.csv
```

字段：

```text
date,tool,model,task,input_scope,output_scope,retained_or_discarded,human_verification,affected_files,notes
```

执行边界：

- AI 可以辅助头脑风暴、语言润色、代码原型和测试设计；
- 最终方法、规则、实验协议和结论由作者确认；
- 所有代码必须经过测试；
- 所有数字必须由脚本从原始日志生成；
- 所有引用必须核对原文或 DOI；
- 不允许生成或补写不存在的实验结果；
- AI 不能列为作者；
- 正式投稿时根据 ECAI 2027 最终政策披露实际使用范围。
