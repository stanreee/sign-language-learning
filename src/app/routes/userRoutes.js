import express from 'express';
const router = express.Router();
import UserController from '../controllers/userController.js';

/* GET users listing. */
router.get('/', UserController.getAllUsers);

router.get('/get-user/', UserController.getOneUser);

router.post('/add-user', UserController.addNewUser);

router.delete('/delete-user/', UserController.deleteUser);

router.post('/auth/', UserController.auth);

router.post('/verify/', UserController.verify);

router.post('/check-account/', UserController.checkAccount);

export default router;
