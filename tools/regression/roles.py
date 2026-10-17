"""What each own repo's model is for, as the catalog regression counts it.

Suite v1 asks chat questions. A bundle whose model is not a chat model (or
that runs on an NPU only) still runs every check, but suite v1 does not apply
to it: `list_bundles.py` writes `role` into every `bundles.json` row from this
map (a repo it does not name is `chat`), and `summarize.py` counts the other
roles apart. Each entry quotes the sentence of the repo's Hub model card that
says what the model is: the first sentence, or the first that names the task
when the first only names the format; numbers left out.

Stdlib only.
"""

from __future__ import annotations

DEFAULT = "chat"
VALUES = (DEFAULT, "classifier", "ocr", "translate", "transform", "decision", "npu-only")

ROLES = {
    # "... so a policy-adaptive safety classifier runs fully on-device ... with no network
    # call." (and: "Shieldstral is not a chat model.")
    "litert-community/Shieldstral-1.0-3B": "classifier",
    # "... converted to the LiteRT-LM (.litertlm) format for on-device document AI ... the
    # first document-parsing / OCR-specialist VLM in this format."
    "litert-community/PaddleOCR-VL-1.6": "ocr",
    # "... converted to the LiteRT-LM (.litertlm) format for on-device document conversion
    # with Google's LiteRT-LM runtime."
    "litert-community/granite-docling-258M": "ocr",
    # "... is a page-level document-parsing model ...: given a document page image it emits
    # a Markdown transcription in reading order"
    "mlboydaisuke/OvisOCR2-LiteRT": "ocr",
    # "... is Tencent's "fast-thinking" multilingual translation model: ... tuned to follow
    # translation instructions rather than chat."
    "litert-community/Hy-MT2-1.8B": "translate",
    # "On-device conversion of ... an Arabic diacritization (tashkeel) fine-tune of ... to
    # .litertlm bundles for the LiteRT-LM runtime."
    "mlboydaisuke/Tashkeel-350M-v2-LiteRT": "transform",
    # "To route a support ticket or score a yes/no question, supply a state and typed
    # questions and read option probabilities."
    "litert-community/decider-0.8b-LiteRT": "decision",
    # "Give it one image and lettered questions, and it returns a probability for each
    # option."
    "litert-community/decider-2b-vision-LiteRT": "decision",
    # "A static-range-quantized (SRQ) .litertlm of ... that a Qualcomm device compiles for
    # its own Hexagon NPU (HTP) at load time (JIT)." A backend rather than a role: CPU and
    # GPU create no engine for it, so suite v1 counts it apart the same way.
    "mlboydaisuke/gemma-3-270m-it-NPU-LiteRT": "npu-only",
}


def role_of(hf_repo: str | None) -> str:
    return ROLES.get(hf_repo or "", DEFAULT)
