# Prompt log

This log records user requests and implementation prompts that guided the local HW2 work. The course assignment brief is included as [ASSIGNMENT.md](ASSIGNMENT.md). No image-generation model prompts were used; all synthetic views and reconstructions were produced by Blender and the repository code.

## User prompts

1. **Original request (Chinese):** “根据E:\HKU-PhD-courses\2503\hw2.md的信息，与当前项目已经包含的文件内容 完成hw2（远程的GitHub项目我会后续建立，你在本地把任务完成即可）”
   - **English translation:** “Using the information in `E:\HKU-PhD-courses\2503\hw2.md` and the files already included in the current project, complete HW2. I will create the remote GitHub project later; complete the work locally for now.”
2. **Continuation:** “你可以继续了” (English: “You can continue.”)
3. **Clarification:** “这些标注为待填写即可，你可以继续了” (English: “Mark these items as TODO to be filled in; you can continue.”)

## Implementation and review prompts

4. **Documentation task:** Complete HW2 locally from the supplied assignment and existing project artifacts. Write the English pipeline report, README, and prompt log. Use actual generated data and metrics; wait for results if needed and do not invent them. Cover ten objects, known synthetic poses, feature matching, NeRF, held-out evaluation, pipeline factors, limitations, reproducibility, and attribution. Keep explicit TODO placeholders for group number, member names/contributions, and future GitHub URL. Verify claims from files.
5. **Review instructions:** Report the initial SIFT 0.8 outcome separately from the selected multi-trial matches. Disclose post-hoc known-pose epipolar QC and unequal candidate-pair counts. Avoid claiming ten successful geometric estimates. Update all result tables from completed metrics and include the eight-versus-32-view ablation when available.

The assignment asks for code, all prompts, matching results, one NeRF/3DGS GIF per object, team details, and a repository URL. This log omits system and developer instructions.
