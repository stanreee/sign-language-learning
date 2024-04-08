import React, { useEffect, useState } from "react";
import Modal from 'react-modal';
import Words from "../ContainerLetters/Words";
import "./WordsModal.css";

const PracticeHelpModal = (props: any) => {
    const {
        isOpen,
        onRequestClose,
        styles,
        title
    } = props;

    return (
        <Modal
            isOpen={isOpen}
            onRequestClose={onRequestClose}
            style={styles}
        >
            <div className="modal-header">
                <h2>
                    {title}
                </h2>
            </div>
            {props.children}
            <div className="modal-footer">
                <button className="Button" onClick={onRequestClose}>Continue</button>
            </div>
        </Modal>
    );
}

export default PracticeHelpModal;