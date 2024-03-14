import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// info from https://clerk.com/blog/building-a-react-login-page-template

const SignUp = (props) => {
    const [name, setName] = useState("")
    const [nameError, setNameError] = useState("")
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [emailError, setEmailError] = useState("")
    const [passwordError, setPasswordError] = useState("")
    const [handedness, setHand] = useState("")
    const [handError, setHandError] = useState("")
    
    const navigate = useNavigate();
        
    const onButtonClick = () => {

        // Set initial error values to empty
        setNameError("")
        setEmailError("")
        setPasswordError("")
        setHandError("")

        // Check if the user has entered both fields correctly
        if ("" === name) {
            setNameError("Please enter your name")
            return
        }
        if ("" === email) {
            setEmailError("Please enter your email")
            return
        }
        if (!/^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$/.test(email)) {
            setEmailError("Please enter a valid email")
            return
        }
        if ("" === password) {
            setPasswordError("Please enter a password")
            return
        }
        if (password.length < 7) {
            setPasswordError("The password must be 8 characters or longer")
            return
        } 
        if ("" === handedness) {
            setHandError("Not selected yet")
            return
        } 

        // Check if email has an account associated with it
        checkAccountExists((accountExists) => {
            // If yes, log in
            if (accountExists) SignUp()
            // Else, ask user if they want to create a new account and if yes, then log in
            else if (
            window.confirm(
                'An account does not exist with this email address: ' + email + '. Do you want to create a new account?',
            )
            ) {
                SignUp()
            navigate("/loggedIn")
            }
        })
    }

    // Call the server API to check if the given email ID already exists
    const checkAccountExists = (callback) => {
        fetch('http://localhost:3080/check-account', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
        })
        .then((r) => r.json())
        .then((r) => {
            callback(r?.userExists)
        })
    }

    // Sign up a user using email and password
    const SignUp = () => {
        fetch('http://localhost:3080/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password, handedness }),
        })
        .then((r) => r.json())
        .then((r) => {
            if ('success' === r.message) {
            localStorage.setItem('user', JSON.stringify({ email, token: r.token }))
            props.setLoggedIn(true)
            props.setEmail(email)
            props.setName(name)
            navigate('/loggedIn')
            } else {
            window.alert('Wrong email or password')
            }
        })
    }

    return <div className={"mainContainer"}>
        <header>
            <h1>Sign Up to ASLingo</h1>
        </header>
        <br />
        <div className={"inputContainer"}>
            <input
                value={name}
                placeholder="Enter your name here"
                onChange={ev => setName(ev.target.value)}
                className="box" />
            <label className="errorLabel">{nameError}</label>
        </div>
        <br />
        <div className={"inputContainer"}>
            <input
                value={email}
                placeholder="Enter your email here"
                onChange={ev => setEmail(ev.target.value)}
                className="box" />
            <label className="errorLabel">{emailError}</label>
        </div>
        <br />
        <div className={"inputContainer"}>
            <input
                type="password"
                value={password}
                placeholder="Enter your password here"
                onChange={ev => setPassword(ev.target.value)}
                className="box" />
            <label className="errorLabel">{passwordError}</label>
        </div>
        <br />
        <div>
            <label className="errorLabel">Choose your handedness: {handedness}</label>
            <div className="container">
                <label className="errorLabel">{handError}</label>
                <input
                    className={"box"}
                    type="button"
                    onClick={ev => setHand(ev.target.value)}
                    value={"Left"} />
                <input
                    className={"box"}
                    type="button"
                    onClick={ev => setHand(ev.target.value)}
                    value={"Right"} />
            </div>
        </div>
        <br />
        <div className={"inputContainer"}>
            <input
                className={"box"}
                type="button"
                onClick={onButtonClick}
                value={"Sign Up"} />
        </div>
    </div>
}

export default SignUp