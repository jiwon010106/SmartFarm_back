const router = require("express").Router();
const { writeCtrl } = require("../controllers/writeCtrl");
const { authenticateToken } = require("../utils/authenticate");

// 게시글 생성
router.post("/create", authenticateToken, writeCtrl.create);

// 모든 게시글 조회
router.get("/", writeCtrl.getPosts);

// 특정 게시글 조회
router.get("/:id", writeCtrl.getPost);

// 게시글 수정
router.put("/:id", authenticateToken, writeCtrl.updatePost);

// 게시글 삭제
router.delete("/:id", authenticateToken, writeCtrl.deletePost);

module.exports = router;
