{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww38200\viewh19980\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # GS-IG: Manifold-Aware Baselines and Group-Sparse Path Optimization for Integrated Gradients\
\
This repository accompanies the ML4H 2025 paper:\
\
> **[Group-Sparse Manifold-Aware Integrated Gradients for Multimodal Transformers on EHR Trajectories]**  \
> Ali Amirahmadi, \
> Proceedings of Machine Learning Research (PMLR) Vol. 297, ML4H 2025.\
\
Integrated Gradients (IG) is a popular method for explaining clinical deep models\'97including widely used multimodal, pretrained Transformers\'97but its utility on EHR code sequences is hampered by (i) the lack of principled baselines for sequence of discrete tokens and (ii) dense, hard-to-interpret generated attributions. To address both, first, we introduce a manifold-aware baseline: the mean input embedding (computed on the validation set), which keeps IG\'92s interpolated points close to typical sequences in embedding space. Second, we introduce \{GS-IG\}, which preserves the straight path geometry but re-parameterizes the schedule \\(\\alpha(t)=t^\{\\theta\}\\) and selects \\(\\theta\\) per input by minimizing a token-level \\(\\ell_\{2,1\}\\) (group-sparsity) objective, producing concise, practitioner-friendly explanations. On MIMIC-IV (incident heart failure) and MDC (early mortality), the manifold-aware baseline improves faithfulness (higher Comprehensiveness, lower Sufficiency), and GS-IG reduces token-level \\(\\ell_\{2,1\}\\) by 9\'9618\\% with negligible change in those metrics on the manifold-aware baseline. The method is lightweight and yields faithful, sparse, and actionable explanations.\
}