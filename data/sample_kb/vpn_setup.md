# Corporate VPN Setup Guide

## Overview
All remote employees must connect to the corporate VPN before accessing internal engineering dashboards, Jira, or staging databases.

## Connection Steps
1. Open the **Cisco AnyConnect Secure Mobility Client** on your laptop.
2. Enter the gateway address: `vpn.internal.company.com`.
3. Click **Connect** and sign in with your corporate SSO email and password.
4. Approve the **Okta Verify** push notification on your mobile device.

## Common Troubleshooting
- **Error: "Certificate validation failure"** — Restart the AnyConnect client and verify your system clock is synced to automatic time.
- **Connection drops every 15 minutes** — Switch from Wi-Fi to wired Ethernet or reconnect on the `Backup-US-East` gateway profile.
- **Locked out after 5 failed attempts** — Wait 15 minutes for automatic unlock, or open a P2 ticket in the IT Service Desk portal.
