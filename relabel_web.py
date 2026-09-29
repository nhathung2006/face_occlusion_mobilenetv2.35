from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config_4class_single_logit.yaml"
MAX_REQUEST_BYTES = 16_384

PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face label review</title>
<style>
:root{color-scheme:dark;--bg:#101722;--panel:#182332;--muted:#9cafc2;--line:#2b3b4d;--accent:#58c4ad;--text:#edf4fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 Segoe UI,Arial,sans-serif}
header{padding:16px 22px;background:#131e2b;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:18px;align-items:center;position:sticky;top:0;z-index:2}
h1{font-size:20px;margin:0}header p{margin:3px 0 0;color:var(--muted);font-size:13px}.layout{max-width:1250px;margin:20px auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.toolbar select,.toolbar input{background:#111b27;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:9px}[hidden]{display:none!important}
.photo{min-height:390px;height:min(62vh,650px);background:#0c131d;border-radius:9px;display:grid;place-items:center;overflow:hidden}.photo img{width:100%;height:100%;object-fit:contain}.empty{color:var(--muted);padding:50px;text-align:center}
.meta{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:var(--muted);margin:12px 0}.path{overflow-wrap:anywhere;font:12px Consolas,monospace;color:#c4d2df}.badge{padding:5px 9px;border-radius:99px;background:#25364a;color:#d7e6f5}.badge.reviewed{background:#174d45;color:#b8f6e6}.choices{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:14px}.btn{border:1px solid var(--line);background:#223247;color:var(--text);padding:12px;border-radius:8px;cursor:pointer;font-weight:650;text-align:left}.btn:hover{border-color:var(--accent);background:#293e52}.btn.primary{background:#176b5b;border-color:#258c77}.btn.warn{background:#654826}.btn.danger{background:#702f3b}.btn:disabled{opacity:.45;cursor:not-allowed}.controls{display:flex;gap:8px;justify-content:space-between;margin-top:12px}.side h2{font-size:15px;margin:0 0 10px}.statgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stat{background:#111b27;padding:10px;border-radius:8px}.stat strong{display:block;font-size:19px}.stat small{color:var(--muted)}.counts{margin:12px 0}.countrow{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line);font-size:13px}.hint{font-size:12px;color:var(--muted);margin-top:12px}.side .btn{width:100%;margin-top:9px}.msg{padding:10px 12px;border-radius:8px;background:#174d45;color:#c7ffeb;font-size:13px;margin-top:12px;overflow-wrap:anywhere}.msg.error{background:#602b34;color:#ffe4e6}.confirm-box{margin-top:12px;padding:12px;border:1px solid #b98546;border-radius:8px;background:#3a3026}.confirm-box p{margin:0 0 10px}.confirm-box .btn{margin-right:8px}.review-label{font-size:13px;margin:4px 0}.completion{padding:12px;border-radius:8px;background:#174d45;color:#c7ffeb;margin-bottom:12px}.completion[hidden]{display:none}.kbd{font:11px Consolas,monospace;border:1px solid #53667a;border-radius:4px;padding:2px 5px;color:#dce9f5}
@media(max-width:850px){.layout{grid-template-columns:1fr}.photo{height:52vh;min-height:260px}header{position:static}}
</style></head><body>
<header><div><h1 id="pageTitle">Rà soát nhãn theo checkpoint mới nhất</h1><p id="pageDescription">Hiển thị ảnh train/val dự đoán sai ở mọi confidence và ảnh dự đoán đúng nhưng confidence &lt; 0,60. Chọn nhãn sẽ cập nhật dataset ngay.</p></div><div id="progress" class="badge">Đang tải…</div></header>
<main class="layout"><section class="panel">
<div class="completion" id="completion" hidden></div>
<div class="toolbar"><button class="btn" id="prev">← Ảnh trước</button><button class="btn" id="next">Ảnh tiếp →</button><select id="splitFilter"><option value="all">Train và val</option><option value="train">Train</option><option value="val">Val</option></select><select id="poseFilter" hidden><option value="priority">Ưu tiên: nghiêng mạnh</option><option value="all">Toàn bộ clear_side_face</option></select><select id="filter"><option value="all">Tất cả nhãn ban đầu</option></select><select id="status"><option value="all">Tất cả trạng thái</option><option value="todo">Chưa rà soát</option><option value="done">Đã rà soát</option></select><input id="search" placeholder="Lọc theo tên/đường dẫn"></div>
<div class="photo" id="photo"><div class="empty">Đang nạp ảnh…</div></div>
<div class="meta"><span id="filename" class="path"></span><span id="statusBadge" class="badge">Chưa rà soát</span></div>
<div class="review-label">Nhãn trong dataset: <b id="original"></b>　→　Nhãn bạn chọn: <b id="assigned"></b></div><div class="review-label">Checkpoint: <b id="checkpointInfo"></b> · Nhãn lúc train: <b id="trueLabel"></b> · Model đoán: <b id="prediction"></b> · <span id="confidenceLabel">Confidence</span>: <b id="confidence"></b> · Lý do rà soát: <b id="reason"></b></div>
<div class="choices" id="choices"></div>
<div class="controls"><button class="btn" id="keep">Giữ nhãn hiện tại <span class="kbd">K</span></button><button class="btn warn" id="reset">Trả về nhãn lúc train</button><button class="btn danger" id="delete">Đưa ảnh khỏi dataset</button></div>
<div class="confirm-box" id="confirmBox" hidden><p id="confirmText"></p><button class="btn danger" id="confirmYes">Xác nhận</button><button class="btn" id="confirmNo">Hủy</button></div>
<div class="msg" id="message" role="status" aria-live="polite" hidden></div>
</section><aside class="panel side"><h2>Tiến độ rà soát</h2><div class="statgrid"><div class="stat"><strong id="reviewedCount">0</strong><small>Đã rà soát</small></div><div class="stat"><strong id="remainingCount">0</strong><small>Còn lại</small></div><div class="stat"><strong id="deletedCount">0</strong><small>Đã cách ly</small></div><div class="stat"><strong id="changedCount">0</strong><small>Đã đổi nhãn</small></div></div><div class="counts" id="counts"></div><p class="hint">Số lượng trên chỉ tính ảnh trong phiên rà soát này, không phải toàn bộ dataset.</p><p class="hint">Phím tắt: <span class="kbd">1–4</span> chọn nhãn, <span class="kbd">K</span> giữ nhãn, <span class="kbd">D</span> cách ly ảnh, <span class="kbd">← →</span> chuyển ảnh.</p><p class="hint">Xóa sẽ chuyển ảnh vào thư mục cách ly <code>_relabel_deleted</code> ngoài bốn thư mục lớp; không xóa vĩnh viễn. Có thể khôi phục bằng cách chọn một nhãn.</p><p class="hint">Danh sách ứng viên được cố định trong manifest để không đổi sau khi di chuyển ảnh.</p></aside></main>
<script>
let state=null,visible=[],position=0,busy=false,previousReviewed=null,completionAlertShown=false,pendingDelete=null;
const $=id=>document.getElementById(id);
async function request(path,opts={}){const r=await fetch(path,opts);const data=await r.json();if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data}
function showMessage(message,error=false){$('message').hidden=false;$('message').className='msg'+(error?' error':'');$('message').textContent=message}
function clearConfirmation(){pendingDelete=null;$('confirmBox').hidden=true}
function getFiltered(){const cls=$('filter').value,status=$('status').value,q=$('search').value.trim().toLowerCase(),split=$('splitFilter').value,priority=state.review_mode==='side_pose'&&$('poseFilter').value==='priority';visible=state.items.filter(x=>(cls==='all'||x.original_label===cls)&&(split==='all'||x.split===split)&&(!priority||x.pose_priority)&&(status==='all'||(status==='todo'&&!x.reviewed)||(status==='done'&&x.reviewed))&&(!q||x.path.toLowerCase().includes(q)));position=Math.min(position,Math.max(0,visible.length-1));render()}
function render(){if(!state)return;const split=$('splitFilter').value,priority=state.review_mode==='side_pose'&&$('poseFilter').value==='priority',scope=state.items.filter(x=>(split==='all'||x.split===split)&&(!priority||x.pose_priority)),reviewed=scope.filter(x=>x.reviewed).length,complete=scope.length>0&&reviewed===scope.length;$('progress').textContent=`${reviewed}/${scope.length} đã rà soát`;$('reviewedCount').textContent=reviewed;$('remainingCount').textContent=scope.length-reviewed;$('deletedCount').textContent=state.deleted_count;$('changedCount').textContent=state.changed_count;$('checkpointInfo').textContent=`epoch ${state.checkpoint_epoch}`;$('completion').hidden=!complete;if(complete)$('completion').textContent=`Đã gán xong! Đã rà soát ${reviewed}/${scope.length} ảnh trong nhóm đang xem; ${state.changed_count} ảnh đã đổi nhãn, ${state.deleted_count} ảnh đã đưa vào cách ly.`;if(complete&&previousReviewed!==null&&previousReviewed<scope.length&&!completionAlertShown){completionAlertShown=true;setTimeout(()=>alert(`Đã gán xong ${scope.length} ảnh trong nhóm đang xem.`),50)}previousReviewed=reviewed;
const counts=$('counts');counts.replaceChildren();for(const label of state.class_names){const row=document.createElement('div');row.className='countrow';row.innerHTML=`<span>${label}</span><b>${state.counts[label]||0}</b>`;counts.append(row)}
const item=visible[position];if(!item){clearConfirmation();$('photo').innerHTML='<div class="empty">Không còn ảnh phù hợp với bộ lọc.</div>';$('filename').textContent='';$('original').textContent='—';$('assigned').textContent='—';$('trueLabel').textContent='—';$('prediction').textContent='—';$('confidence').textContent='—';$('reason').textContent='—';$('statusBadge').textContent='';$('choices').replaceChildren();$('delete').textContent='Đưa ảnh khỏi dataset';$('delete').onclick=null;return}
if(pendingDelete&&pendingDelete.id!==item.id)clearConfirmation();
$('filename').textContent=`${position+1}/${visible.length} · [${item.split}] ${item.path}`;$('original').textContent=item.deleted?'Đã cách ly':item.current_label;$('assigned').textContent=item.deleted?'—':item.current_label;$('trueLabel').textContent=item.original_label;$('prediction').textContent=item.predicted_label;$('confidence').textContent=Number(state.review_mode==='side_pose'?item.pose_probability:item.confidence).toFixed(3);$('reason').textContent=({misclassified_and_low_confidence:'Dự đoán sai, confidence thấp',misclassified_high_confidence:'Dự đoán sai, confidence cao',correct_low_confidence:'Dự đoán đúng, confidence thấp',pose_priority:'Nghiêng mạnh: cần kiểm tra',pose_other:'Toàn bộ clear_side_face'})[item.review_reason]||'—';$('statusBadge').textContent=item.deleted?'Đã cách ly':item.reviewed?'Đã rà soát':'Chưa rà soát';$('statusBadge').className='badge '+(item.reviewed?'reviewed':'');$('photo').innerHTML=`<img alt="Ảnh cần rà soát" src="/api/image?id=${item.id}&v=${Date.now()}">`;
const choices=$('choices');choices.replaceChildren();state.class_names.forEach((label,i)=>{const b=document.createElement('button');b.className='btn '+(label===item.target_label?'primary':'');b.textContent=`[${i+1}] ${label}`;b.onclick=()=>assign(label);choices.append(b)});
$('delete').textContent=item.deleted?'Khôi phục ảnh vào dataset':'Đưa ảnh khỏi dataset';$('delete').onclick=()=>deleteOrRestore(item);
}
async function refresh(keepPath=true){const old=keepPath&&visible[position]?visible[position].id:null;state=await request('/api/state');const poseMode=state.review_mode==='side_pose';$('poseFilter').hidden=!poseMode;$('filter').hidden=poseMode;$('pageTitle').textContent=poseMode?'Rà soát góc mặt: clear_side_face':'Rà soát nhãn theo checkpoint mới nhất';$('pageDescription').textContent=poseMode?`Ưu tiên ảnh có điểm occluded_pose ≥ ${state.priority_probability_min.toFixed(2)}. Đây là điểm gợi ý; hãy xem ảnh trước khi gán nhãn. Có thể chuyển sang toàn bộ clear_side_face ở train và val.`:'Hiển thị ảnh train/val dự đoán sai ở mọi confidence và ảnh dự đoán đúng nhưng confidence < 0,60. Chọn nhãn sẽ cập nhật dataset ngay.';$('confidenceLabel').textContent=poseMode?'Điểm occluded_pose':'Confidence';if($('filter').options.length===1){for(const c of state.class_names){const o=document.createElement('option');o.value=c;o.textContent=`Nhãn lúc train: ${c}`;$('filter').append(o)}}getFiltered();let idx=old===null?-1:visible.findIndex(x=>x.id===old);if(idx>=0)position=idx;render()}
async function assign(label){if(busy||!visible[position])return;busy=true;clearConfirmation();try{const item=visible[position],todoFilter=$('status').value==='todo';const result=await request('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,target_label:label})});await refresh(false);showMessage(label===item.current_label?'Đã xác nhận giữ nhãn; tệp trong dataset không đổi.':`Đã đổi nhãn trong dataset: ${result.path}`);if(!todoFilter&&visible.some(x=>x.id===item.id)&&position<visible.length-1)position++;render()}catch(e){showMessage(`Không thể gán nhãn: ${e.message}`,true)}finally{busy=false}}
function move(delta){position=Math.max(0,Math.min(visible.length-1,position+delta));render()}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('filter').onchange=()=>{position=0;getFiltered()};$('splitFilter').onchange=()=>{position=0;completionAlertShown=false;previousReviewed=null;getFiltered()};$('poseFilter').onchange=()=>{position=0;completionAlertShown=false;previousReviewed=null;getFiltered()};$('status').onchange=()=>{position=0;getFiltered()};$('search').oninput=()=>{position=0;getFiltered()};$('keep').onclick=()=>visible[position]&&assign(visible[position].current_label);$('reset').onclick=()=>visible[position]&&assign(visible[position].original_label);$('delete').onclick=()=>visible[position]&&deleteOrRestore(visible[position]);
function deleteOrRestore(item){if(busy)return;if(pendingDelete&&pendingDelete.id===item.id&&pendingDelete.deleted===item.deleted){clearConfirmation();return}pendingDelete={id:item.id,deleted:item.deleted};$('confirmText').textContent=item.deleted?'Khôi phục ảnh này vào thư mục lớp trong dataset?':'Đưa ảnh này khỏi bốn lớp? Ảnh sẽ được chuyển vào _relabel_deleted và có thể khôi phục.';$('confirmYes').textContent=item.deleted?'Xác nhận khôi phục':'Xác nhận cách ly';$('confirmBox').hidden=false}
$('confirmNo').onclick=clearConfirmation;
$('confirmYes').onclick=async()=>{if(!pendingDelete||busy)return;const {id,deleted}=pendingDelete;clearConfirmation();await action(deleted?'/api/restore':'/api/delete',id,deleted?'Đã khôi phục ảnh trong dataset':'Đã đưa ảnh khỏi bốn lớp vào thư mục cách ly')};
async function action(url,id,message){busy=true;try{const todoFilter=$('status').value==='todo';const result=await request(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});await refresh(false);showMessage(`${message}: ${result.path}`);if(!todoFilter&&position<visible.length-1)position++;render()}catch(e){showMessage(`Không thể cập nhật dataset: ${e.message}`,true)}finally{busy=false}}
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key>='1'&&e.key<='4'&&state)assign(state.class_names[Number(e.key)-1]);else if(e.key.toLowerCase()==='k')$('keep').click();else if(e.key.toLowerCase()==='d')$('delete').click();else if(e.key==='ArrowRight')move(1);else if(e.key==='ArrowLeft')move(-1)});
refresh(false).catch(e=>showMessage(e.message,true));
</script></body></html>"""


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_app_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for section in ("data", "relabel_web"):
        if section not in cfg:
            raise ValueError(f"Missing {section!r} section in {config_path}")
    labels = cfg["data"].get("class_names", [])
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("data.class_names must contain unique labels.")
    return cfg


class RelabelService:
    def __init__(self, config: dict, config_path: Path, review_mode: str = "errors"):
        if review_mode not in ("errors", "side_pose"):
            raise ValueError(f"Unsupported review mode: {review_mode}")
        self.review_mode = review_mode
        self.config_path = config_path.resolve()
        self.data_root = resolve_project_path(config["data"]["root"])
        self.class_names = list(config["data"]["class_names"])
        self.web_cfg = config["relabel_web"]
        side_cfg = self.web_cfg.get("side_pose_review", {})
        if review_mode == "side_pose":
            source_label = side_cfg.get("source_label", "clear_side_face")
            pose_label = side_cfg.get("pose_label", "occluded_pose")
            if source_label not in self.class_names or pose_label not in self.class_names:
                raise ValueError("side_pose_review labels must belong to data.class_names.")
            priority_min = float(side_cfg.get("priority_probability_min", 0.20))
            if not 0.0 <= priority_min <= 1.0:
                raise ValueError("side_pose_review.priority_probability_min must be in [0, 1].")
            self.filter_cfg = {
                "review_mode": "side_pose", "source_label": source_label,
                "pose_label": pose_label, "priority_probability_min": priority_min,
                "splits": ["train", "val"], "prediction_head": "subtype_4class",
            }
            manifest_base = resolve_project_path(
                side_cfg.get("manifest_path", "outputs/relabel_review/clear_side_pose_review.json")
            )
        else:
            self.filter_cfg = self.web_cfg["review_filter"]
            manifest_base = resolve_project_path(self.web_cfg["manifest_path"])
        checkpoint_path = resolve_project_path(
            Path(config["paths"]["checkpoint_dir"]) / config["paths"]["best_checkpoint_name"]
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Latest best checkpoint not found: {checkpoint_path}")
        checkpoint_hash = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checkpoint_hash.update(chunk)
        self.checkpoint_sha256 = checkpoint_hash.hexdigest()
        filter_payload = json.dumps(self.filter_cfg, sort_keys=True, separators=(",", ":"))
        filter_hash = hashlib.sha256(filter_payload.encode("utf-8")).hexdigest()[:8]
        self.manifest_path = manifest_base.with_name(
            f"{manifest_base.stem}_{self.checkpoint_sha256[:12]}_{filter_hash}{manifest_base.suffix}"
        )
        self.move_immediately = bool(self.web_cfg["move_images_immediately"])
        self.quarantine_dir = (self.data_root / self.web_cfg.get("quarantine_dir", "_relabel_deleted")).resolve()
        if self.quarantine_dir.parent != self.data_root or self.quarantine_dir.name in self.class_names:
            raise ValueError("relabel_web.quarantine_dir must be a non-class direct child of data.root.")
        self.lock = threading.RLock()
        self.items: list[dict] = []
        if self.manifest_path.is_file():
            self.session = self._load_manifest()
            self.items = self._restore_candidates(self.session["candidates"])
        else:
            self.session, self.items = self._find_candidates()
            self._save_manifest()

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.session["candidates"] = [self._serializable_item(item) for item in self.items]
        temp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self.session, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self.manifest_path)

    @staticmethod
    def _serializable_item(item: dict) -> dict:
        return {key: value for key, value in item.items() if key != "absolute_path"}

    def _load_manifest(self) -> dict:
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            session = json.load(stream)
        if session.get("version") != 2:
            raise ValueError(
                "This manifest is not a filtered train/val review session. "
                "The older decisions.json was preserved; use the configured new manifest path."
            )
        if session.get("data_root") != str(self.data_root) or session.get("class_names") != self.class_names:
            raise ValueError("Review manifest does not match the configured dataset and labels.")
        if session.get("review_filter") != self.filter_cfg:
            raise ValueError("Review manifest does not match the selected filter or mode.")
        return session

    def _restore_candidates(self, records: list[dict]) -> list[dict]:
        items = []
        for record in records:
            if record.get("original_label") not in self.class_names or record.get("current_label") not in self.class_names:
                raise ValueError(f"Invalid label in review manifest for {record.get('candidate_id')}")
            relative = Path(record["current_path"])
            path = (self.data_root / relative).resolve()
            try:
                path.relative_to(self.data_root)
            except ValueError as exc:
                raise ValueError("Manifest image path escapes the dataset root.") from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"A candidate image recorded in the review manifest is missing: {path}"
                )
            deleted = bool(record.get("deleted", False))
            expected_root = self.quarantine_dir if deleted else self.data_root / record["current_label"]
            try:
                path.relative_to(expected_root)
            except ValueError as exc:
                raise ValueError(f"Manifest status and current image folder differ: {path}") from exc
            items.append({**record, "deleted": deleted, "id": len(items), "absolute_path": path})
        return items

    def _find_candidates(self) -> tuple[dict, list[dict]]:
        import train_4class_single_logit as trainer
        from torch.utils.data import DataLoader

        cfg = trainer.load_config(self.config_path)
        if Path(cfg["data"]["root"]).resolve() != self.data_root:
            raise ValueError("The web and training configs resolve to different dataset roots.")
        checkpoint_path = resolve_project_path(
            Path(cfg["paths"]["checkpoint_dir"]) / cfg["paths"]["best_checkpoint_name"]
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Previous-training best checkpoint not found: {checkpoint_path}")
        checkpoint = trainer.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("class_names") != self.class_names:
            raise ValueError("Checkpoint subclass order does not match the four configured folders.")

        train_samples, val_samples, split_counts = trainer.split_samples(cfg)
        saved_counts = checkpoint.get("split_counts")
        if saved_counts is not None and saved_counts != split_counts:
            raise ValueError(
                "Current files/config no longer reproduce the checkpoint's train/val split. "
                "Refusing to guess which images belonged to the previous run."
            )
        model = trainer.build_model(cfg, load_pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        device = trainer.torch.device(str(cfg["runtime"]["evaluation_device"]))
        if device.type == "cuda" and not trainer.torch.cuda.is_available():
            raise RuntimeError("Configured evaluation device is CUDA but CUDA is unavailable.")
        model.to(device).eval()
        _, eval_transform = trainer.make_transforms(cfg)
        class_to_binary = trainer.subclass_to_binary_ids(cfg)
        class_names = list(self.class_names)
        side_mode = self.review_mode == "side_pose"
        threshold = (None if side_mode else float(self.filter_cfg["confidence_max_exclusive"]))
        source_index = (class_names.index(self.filter_cfg["source_label"]) if side_mode else None)
        pose_index = (class_names.index(self.filter_cfg["pose_label"]) if side_mode else None)
        priority_min = (float(self.filter_cfg["priority_probability_min"]) if side_mode else None)
        filter_splits = set(self.filter_cfg["splits"])
        if self.filter_cfg["prediction_head"] != "subtype_4class":
            raise ValueError("The review filter currently supports prediction_head: subtype_4class only.")

        candidates: list[dict] = []
        split_sets = (("train", train_samples), ("val", val_samples))
        with trainer.torch.inference_mode():
            for split_name, samples in split_sets:
                if split_name not in filter_splits:
                    continue
                if side_mode:
                    samples = [(path, label) for path, label in samples if label == source_index]
                loader = DataLoader(
                    trainer.FaceClassDataset(samples, eval_transform),
                    batch_size=int(cfg["evaluation"]["batch_size"]),
                    shuffle=False,
                    num_workers=0,
                )
                offset = 0
                for images, labels in loader:
                    _, subtype_logits = model(images.to(device))
                    probabilities = trainer.torch.softmax(subtype_logits, dim=1)
                    confidences, predictions = probabilities.max(dim=1)
                    for index, true_index in enumerate(labels.tolist()):
                        predicted_index = int(predictions[index])
                        confidence = float(confidences[index])
                        pose_probability = float(probabilities[index, pose_index]) if side_mode else None
                        is_wrong = predicted_index != true_index
                        if side_mode:
                            include = True
                            is_low_confidence = False
                        else:
                            is_low_confidence = confidence < threshold
                            selection = self.filter_cfg.get("selection", "misclassified_and_low_confidence")
                            if selection == "misclassified_or_low_confidence":
                                include = is_wrong or is_low_confidence
                            elif selection == "misclassified_and_low_confidence":
                                include = is_wrong and is_low_confidence
                            else:
                                raise ValueError(f"Unsupported review_filter.selection: {selection}")
                        if not include:
                            continue
                        source_path = samples[offset + index][0]
                        current_rel = source_path.relative_to(self.data_root).as_posix()
                        current_label = class_names[true_index]
                        candidates.append({
                            "candidate_id": trainer.file_sha256(source_path),
                            "current_path": current_rel,
                            "original_path": current_rel,
                            "original_label": current_label,
                            "current_label": current_label,
                            "split": split_name,
                            "predicted_label": class_names[predicted_index],
                            "confidence": confidence,
                            "pose_probability": pose_probability,
                            "pose_priority": bool(side_mode and pose_probability >= priority_min),
                            "review_reason": (
                                ("pose_priority" if pose_probability >= priority_min else "pose_other")
                                if side_mode else
                                "misclassified_and_low_confidence" if is_wrong and is_low_confidence
                                else "misclassified_high_confidence" if is_wrong
                                else "correct_low_confidence"
                            ),
                            "reviewed": False,
                            "deleted": False,
                            "target_label": current_label,
                            "updated_at": None,
                        })
                    offset += len(labels)
        if not candidates:
            raise ValueError("No images match the selected review mode and dataset split.")
        if side_mode:
            candidates.sort(key=lambda item: (-item["pose_probability"], item["current_path"]))
        session = {
            "version": 2,
            "data_root": str(self.data_root),
            "class_names": self.class_names,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint_val_binary_f1": checkpoint.get("val_macro_f1"),
            "review_filter": self.filter_cfg,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "audit": [],
            "candidates": [],
        }
        return session, [{**candidate, "id": index, "absolute_path": self.data_root / candidate["current_path"]}
                         for index, candidate in enumerate(candidates)]

    def item_for_id(self, item_id: int) -> dict:
        if item_id < 0 or item_id >= len(self.items):
            raise IndexError("Image index is out of range.")
        return self.items[item_id]

    def item_result(self, item_id: int) -> dict:
        with self.lock:
            item = self.item_for_id(item_id)
            if not item["absolute_path"].is_file():
                raise FileNotFoundError(f"Updated image is missing: {item['absolute_path']}")
            return {"ok": True, "path": item["current_path"],
                    "current_label": item["current_label"], "deleted": bool(item["deleted"])}

    def state(self) -> dict:
        with self.lock:
            counts = {name: 0 for name in self.class_names}
            reviewed = 0
            items = []
            for item in self.items:
                if not item.get("deleted", False):
                    counts[item["current_label"]] += 1
                is_reviewed = bool(item.get("reviewed", False))
                reviewed += int(is_reviewed)
                items.append({
                    "id": item["id"],
                    "path": item["current_path"],
                    "original_label": item["original_label"],
                    "current_label": item["current_label"],
                    "target_label": item["current_label"],
                    "split": item["split"],
                    "predicted_label": item["predicted_label"],
                    "confidence": item["confidence"],
                    "pose_probability": item.get("pose_probability"),
                    "pose_priority": bool(item.get("pose_priority", False)),
                    "review_reason": item.get("review_reason"),
                    "reviewed": is_reviewed,
                    "deleted": bool(item.get("deleted", False)),
                })
            return {
                "data_root": str(self.data_root),
                "class_names": self.class_names,
                "items": items,
                "counts": counts,
                "reviewed": reviewed,
                "deleted_count": sum(bool(item.get("deleted", False)) for item in self.items),
                "changed_count": sum(item["current_label"] != item["original_label"] for item in self.items),
                "checkpoint_epoch": self.session["checkpoint_epoch"],
                "checkpoint_path": self.session["checkpoint_path"],
                "review_filter": self.session["review_filter"],
                "review_mode": self.review_mode,
                "priority_probability_min": (
                    float(self.filter_cfg["priority_probability_min"])
                    if self.review_mode == "side_pose" else None
                ),
            }

    def set_label(self, item_id: int, target_label: str) -> None:
        if target_label not in self.class_names:
            raise ValueError(f"Unknown target label: {target_label}")
        with self.lock:
            item = self.item_for_id(item_id)
            source = item["absolute_path"]
            old_label = item["current_label"]
            old_rel = item["current_path"]
            destination = source
            new_rel = old_rel
            was_deleted = bool(item.get("deleted", False))
            if self.move_immediately and (target_label != old_label or was_deleted):
                relative_inside_class = (Path(old_rel).relative_to(self.quarantine_dir.relative_to(self.data_root) / old_label)
                                         if was_deleted else Path(old_rel).relative_to(old_label))
                destination = self.data_root / target_label / relative_inside_class
                if destination.exists():
                    raise FileExistsError(
                        f"Cannot relabel without overwriting an existing file: {destination}"
                    )
                if not source.is_file():
                    raise FileNotFoundError(f"Candidate image no longer exists: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                new_rel = destination.relative_to(self.data_root).as_posix()

            old_record = dict(item)
            old_record.pop("absolute_path", None)
            previous_audit_length = len(self.session["audit"])
            item["current_path"] = new_rel
            item["absolute_path"] = destination
            item["current_label"] = target_label if self.move_immediately else old_label
            item["deleted"] = False
            item["target_label"] = target_label
            item["reviewed"] = True
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.session["audit"].append({
                "timestamp": item["updated_at"],
                "candidate_id": item["candidate_id"],
                "split": item["split"],
                "original_label": item["original_label"],
                "old_dataset_label": old_label,
                "new_dataset_label": item["current_label"],
                "model_prediction": item["predicted_label"],
                "confidence": item["confidence"],
                "old_path": old_rel,
                "new_path": new_rel,
                "action": "restore_and_relabel" if was_deleted and target_label != old_label else "restore" if was_deleted else "label",
            })
            try:
                self._save_manifest()
            except Exception:
                del self.session["audit"][previous_audit_length:]
                item.clear()
                item.update(old_record)
                item["absolute_path"] = source
                if destination != source and destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                raise

    def delete_candidate(self, item_id: int) -> None:
        with self.lock:
            item = self.item_for_id(item_id)
            if item.get("deleted", False):
                return
            source = item["absolute_path"]
            if not source.is_file():
                raise FileNotFoundError(f"Candidate image no longer exists: {source}")
            old_record = self._serializable_item(item)
            old_rel = item["current_path"]
            relative_inside_class = Path(old_rel).relative_to(item["current_label"])
            destination = self.quarantine_dir / item["current_label"] / relative_inside_class
            if destination.exists():
                raise FileExistsError(f"Cannot quarantine without overwriting an existing file: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            new_rel = destination.relative_to(self.data_root).as_posix()
            previous_audit_length = len(self.session["audit"])
            item.update({"current_path": new_rel, "absolute_path": destination, "deleted": True,
                         "reviewed": True, "updated_at": datetime.now(timezone.utc).isoformat()})
            self.session["audit"].append({"timestamp": item["updated_at"], "candidate_id": item["candidate_id"],
                "split": item["split"], "action": "quarantine", "old_path": old_rel, "new_path": new_rel,
                "current_label": item["current_label"], "model_prediction": item["predicted_label"],
                "confidence": item["confidence"]})
            try:
                self._save_manifest()
            except Exception:
                del self.session["audit"][previous_audit_length:]
                item.clear()
                item.update(old_record)
                item["absolute_path"] = source
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                raise

    def restore_candidate(self, item_id: int) -> None:
        with self.lock:
            item = self.item_for_id(item_id)
            if not item.get("deleted", False):
                return
            source = item["absolute_path"]
            if not source.is_file():
                raise FileNotFoundError(f"Quarantined image no longer exists: {source}")
            relative_inside_class = Path(item["current_path"]).relative_to(
                self.quarantine_dir.relative_to(self.data_root) / item["current_label"])
            destination = self.data_root / item["current_label"] / relative_inside_class
            if destination.exists():
                raise FileExistsError(f"Cannot restore without overwriting an existing file: {destination}")
            old_record = self._serializable_item(item)
            old_rel = item["current_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            new_rel = destination.relative_to(self.data_root).as_posix()
            previous_audit_length = len(self.session["audit"])
            item.update({"current_path": new_rel, "absolute_path": destination, "deleted": False,
                         "reviewed": True, "updated_at": datetime.now(timezone.utc).isoformat()})
            self.session["audit"].append({"timestamp": item["updated_at"], "candidate_id": item["candidate_id"],
                "split": item["split"], "action": "restore", "old_path": old_rel, "new_path": new_rel,
                "current_label": item["current_label"]})
            try:
                self._save_manifest()
            except Exception:
                del self.session["audit"][previous_audit_length:]
                item.clear()
                item.update(old_record)
                item["absolute_path"] = source
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(source)
                raise


def make_handler(service: RelabelService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FaceRelabelWeb/1.0"

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large.")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/state":
                self._json(200, service.state())
                return
            if parsed.path == "/api/image":
                try:
                    item_id = int(parse_qs(parsed.query).get("id", [""])[0])
                    item = service.item_for_id(item_id)
                    image_path = item["absolute_path"]
                    if not image_path.is_file():
                        raise FileNotFoundError("Image no longer exists.")
                    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(image_path.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with image_path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile)
                except (ValueError, IndexError, FileNotFoundError) as exc:
                    self._json(404, {"error": str(exc)})
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/label":
                    item_id = int(payload["id"])
                    service.set_label(item_id, str(payload["target_label"]))
                    self._json(200, service.item_result(item_id))
                elif parsed.path == "/api/delete":
                    item_id = int(payload["id"])
                    service.delete_candidate(item_id)
                    self._json(200, service.item_result(item_id))
                elif parsed.path == "/api/restore":
                    item_id = int(payload["id"])
                    service.restore_candidate(item_id)
                    self._json(200, service.item_result(item_id))
                else:
                    self._json(404, {"error": "Not found"})
            except FileExistsError as exc:
                self._json(409, {"error": str(exc)})
            except PermissionError:
                self._json(403, {"error": "Web không có quyền ghi vào dataset. Hãy dừng tiến trình web cũ và chạy lại từ terminal VS Code với quyền ghi vào thư mục data/class."})
            except (ValueError, IndexError, KeyError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"[web] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Local web app for manually reviewing four face labels.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("errors", "side-pose"), default="errors",
                        help="Review model errors or clear_side_face pose candidates.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_app_config(config_path)
    service = RelabelService(config, config_path,
                             review_mode="side_pose" if args.mode == "side-pose" else "errors")
    host = args.host or str(config["relabel_web"]["host"])
    port = args.port or int(config["relabel_web"]["port"])
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"Dataset: {service.data_root}")
    reason_counts: dict[str, int] = {}
    for item in service.items:
        reason = str(item.get("review_reason", "previously_reviewed"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    print(f"Review mode: {args.mode}")
    print(f"Eligible images for train + val review: {len(service.items)}")
    print(f"Review groups: {reason_counts}")
    print(f"Checkpoint: {service.session['checkpoint_path']} (epoch {service.session['checkpoint_epoch']})")
    print(f"Manifest: {service.manifest_path}")
    if service.review_mode == "side_pose":
        print(f"Pose priority threshold: >= {service.filter_cfg['priority_probability_min']}")
        print(f"Priority candidates: {sum(item['pose_priority'] for item in service.items)}")
    else:
        print(f"Confidence threshold: < {service.filter_cfg['confidence_max_exclusive']}")
    print(f"Open in browser: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping relabel review server…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
