import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// info from https://clerk.com/blog/building-a-react-login-page-template

const Login = (props) => {
    const [name, setName] = useState("")
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [emailError, setEmailError] = useState("")
    const [passwordError, setPasswordError] = useState("")
    
    const navigate = useNavigate();
        
    const onButtonClick = () => {

        // Set initial error values to empty
        setEmailError("")
        setPasswordError("")

        // Check if the user has entered both fields correctly
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

        // Check if email has an account associated with it
        checkAccountExists((accountExists) => {
            // If yes, log in
            if (accountExists) logIn()
            // Else, tell user to use the "Sign Up" option instead
            else if (
            window.confirm(
                'An account does not exist with this email address: ' + email + '. Please Sign Up!',
            )
            ) {
            navigate("/signup")
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

    // Log in a user using email and password
    const logIn = () => {
        fetch('http://localhost:3080/auth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password}),
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

    return (
        <div className={"mainContainer"}>
            <header>
                <h1>Login to ASLingo</h1>
            </header>
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
            <div className={"inputContainer"}>
                <input
                    className={"box"}
                    type="button"
                    onClick={onButtonClick}
                    value={"Log In"} />
            </div>
        </div>
    )
}

export default Login